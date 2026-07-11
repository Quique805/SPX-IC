#!/usr/bin/env python3
"""Build ML Shadow V1 paper predictions for the 0DTE volatility-selling system.

The shadow model is intentionally dependency-free. It combines:
- candidate generation around SpotGamma / GEX / DEX levels,
- a transparent heuristic no-touch score,
- a small regularized logistic regression implemented in pure Python,
- a small gradient-boosting model made of decision stumps.

It writes JSON only. It never changes the production signal logic.
"""
import json
import math
import os
from datetime import datetime, timezone

from fetch_daily import DATA_DIR, load_json_safe, save_json

VERSION = "ml-shadow-v1.1"
STRIKE_STEP = 5
SPREAD_WIDTH = 20
PROTECTION_WIDTHS = (5, 10, 15, 20, 25, 30, 40)
MIN_REASONABLE_CREDIT_USD = 5.0
MIN_TRAIN_DAYS = 8

ML_DATASET_DIR = os.path.join(DATA_DIR, "ml-dataset")
CHAINS_DIR = os.path.join(DATA_DIR, "chains")
SIGNALS_DIR = os.path.join(DATA_DIR, "signals")
OUT_DIR = os.path.join(DATA_DIR, "ml-shadow")
OUT_PREDICTIONS_DIR = os.path.join(OUT_DIR, "predictions")
OUT_INDEX = os.path.join(OUT_DIR, "index.json")

FEATURE_COLUMNS = [
    "side_call",
    "distance_open_points",
    "distance_open_pct",
    "distance_wall_points",
    "distance_benchmark_points",
    "wall_range",
    "open_pct_walls",
    "wing_credit_usd",
    "sell_bid_usd",
    "sell_mid_usd",
    "buy_ask_usd",
    "spread_pct",
    "iv",
    "abs_delta",
    "volume_log",
    "oi_log",
    "tradable_score",
    "is_dry",
    "gex_distance_side_consensus",
    "gex_side_dispersion",
    "gex_side_agreement",
    "gex_net_positive_count",
    "dex_distance_side_wall",
    "dex_distance_flip",
    "dex_net_positive",
    "front_atm_iv",
    "front_skew_balance",
    "front_minus_second",
    "surface_missing_iv_pct",
    "vix_open",
    "gap_pct",
    "range_pct_5d",
    "realized_vol_5d",
    "trend_5d",
]


def safe_num(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def clamp(value, low, high):
    value = safe_num(value, low)
    return max(low, min(high, value))


def logistic(x):
    if x >= 35:
        return 1.0
    if x <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def round_step(value, step=STRIKE_STEP):
    value = safe_num(value)
    if value is None:
        return None
    return int(round(value / step) * step)


def strike_range(start, end, step=STRIKE_STEP):
    start = round_step(start, step)
    end = round_step(end, step)
    if start is None or end is None:
        return []
    if start > end:
        start, end = end, start
    return list(range(start, end + step, step))


def mean(values):
    clean = [safe_num(v) for v in values]
    clean = [v for v in clean if v is not None]
    return sum(clean) / len(clean) if clean else None


def stdev(values):
    clean = [safe_num(v) for v in values]
    clean = [v for v in clean if v is not None]
    if len(clean) < 2:
        return None
    avg = sum(clean) / len(clean)
    return math.sqrt(sum((v - avg) ** 2 for v in clean) / (len(clean) - 1))


def log1p(value):
    value = safe_num(value, 0)
    return math.log1p(max(0.0, value))


def load_rows(name, key="date"):
    payload = load_json_safe(os.path.join(ML_DATASET_DIR, f"{name}.json")) or {}
    rows = payload.get("rows") or []
    out = {}
    for row in rows:
        row_key = row.get(key) or row.get("targetSession")
        if row_key:
            out[row_key] = row
    return out


def option_expiration(chain, date_str):
    expirations = chain.get("expirations") or {}
    if date_str in expirations:
        return date_str, expirations[date_str]
    usable = []
    for exp, data in expirations.items():
        dte = safe_num(data.get("dte"))
        if dte is not None and dte >= 0:
            usable.append((dte, exp, data))
    if not usable:
        return None, None
    usable.sort(key=lambda item: (item[0], item[1]))
    return usable[0][1], usable[0][2]


def nearest_strike_row(expiration_data, strike):
    rows = expiration_data.get("strikes") or []
    if not rows:
        return None
    strike = safe_num(strike)
    if strike is None:
        return None
    return min(rows, key=lambda row: abs(safe_num(row.get("strike"), 10**9) - strike))


def side_snapshot(row, side):
    if not row:
        return {}
    prefix = "call" if side == "call" else "put"
    bid = safe_num(row.get(f"{prefix}_bid"))
    ask = safe_num(row.get(f"{prefix}_ask"))
    last = safe_num(row.get(f"{prefix}_last"))
    mid = None
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2
    elif last is not None:
        mid = last
    spread = ask - bid if bid is not None and ask is not None else None
    spread_pct = spread / mid * 100 if spread is not None and mid and mid > 0 else None
    volume = safe_num(row.get(f"{prefix}_volume"), 0)
    oi = safe_num(row.get(f"{prefix}_oi"), 0)
    iv = safe_num(row.get(f"{prefix}_iv"))
    delta = safe_num(row.get(f"{prefix}_delta"))
    premium_usd = bid * 100 if bid is not None else None
    mid_usd = mid * 100 if mid is not None else None
    liquidity = math.log1p(max(0, volume)) + 0.35 * math.log1p(max(0, oi))
    premium_component = math.log1p(max(0, premium_usd or 0))
    spread_penalty = 1 + max(0, spread_pct or 0) / 35
    tradable = liquidity * premium_component / spread_penalty
    is_dry = bool(
        premium_usd is None
        or premium_usd <= MIN_REASONABLE_CREDIT_USD
        or bid is None
        or bid <= 0
        or (spread_pct is not None and spread_pct > 100)
    )
    return {
        "strike": safe_num(row.get("strike")),
        "bid": bid,
        "ask": ask,
        "last": last,
        "mid": mid,
        "spread": spread,
        "spreadPct": spread_pct,
        "iv": iv,
        "delta": delta,
        "absDelta": abs(delta) if delta is not None else None,
        "volume": volume,
        "openInterest": oi,
        "premiumUsd": premium_usd,
        "midPremiumUsd": mid_usd,
        "tradableScore": tradable,
        "isDry": is_dry,
    }


def wing_snapshot(chain, date_str, side, sell_strike, width=SPREAD_WIDTH):
    exp_key, exp_data = option_expiration(chain, date_str)
    if not exp_data:
        return {}
    buy_strike = sell_strike + width if side == "call" else sell_strike - width
    sell_row = nearest_strike_row(exp_data, sell_strike)
    buy_row = nearest_strike_row(exp_data, buy_strike)
    sell = side_snapshot(sell_row, side)
    buy = side_snapshot(buy_row, side)
    actual_buy_strike = safe_num(buy.get("strike"))
    if actual_buy_strike is not None:
        if side == "call" and actual_buy_strike <= sell_strike:
            buy = {}
            actual_buy_strike = None
        if side == "put" and actual_buy_strike >= sell_strike:
            buy = {}
            actual_buy_strike = None
    actual_width = abs(actual_buy_strike - sell_strike) if actual_buy_strike is not None else width
    sell_bid = safe_num(sell.get("bid"))
    buy_ask = safe_num(buy.get("ask"))
    credit = sell_bid - buy_ask if sell_bid is not None and buy_ask is not None else None
    credit_usd = credit * 100 if credit is not None else None
    return {
        "expiration": exp_key,
        "dte": safe_num(exp_data.get("dte")),
        "sell": sell,
        "buy": buy,
        "buyStrike": buy_strike,
        "buyStrikeActual": actual_buy_strike,
        "spreadWidthPoints": actual_width,
        "wingCredit": credit,
        "wingCreditUsd": credit_usd,
        "maxRiskUsd": actual_width * 100 - credit_usd if credit_usd is not None else None,
        "reasonableCredit": credit_usd is not None and credit_usd > MIN_REASONABLE_CREDIT_USD,
    }


def protection_options(chain, date_str, side, sell_strike):
    options = []
    for width in PROTECTION_WIDTHS:
        snap = wing_snapshot(chain, date_str, side, sell_strike, width=width)
        credit_usd = safe_num(snap.get("wingCreditUsd"))
        max_risk = safe_num(snap.get("maxRiskUsd"))
        if credit_usd is None or max_risk is None or max_risk <= 0:
            continue
        options.append({
            "width": safe_num(snap.get("spreadWidthPoints"), width),
            "buyStrike": snap.get("buyStrikeActual") or snap.get("buyStrike"),
            "creditUsd": credit_usd,
            "maxRiskUsd": max_risk,
            "riskRewardPct": credit_usd / max_risk * 100 if max_risk > 0 else None,
            "reasonableCredit": credit_usd > MIN_REASONABLE_CREDIT_USD,
        })
    return options


def protection_target(candidate):
    prob = safe_num(candidate.get("finalProbNoTouch"), candidate.get("heuristicProbNoTouch")) or 0.5
    distance = safe_num(candidate.get("distanceOpenPoints"), 0)
    delta = safe_num(candidate.get("absDelta"), 0.25)
    risk = (
        0.55 * (1 - clamp(prob, 0, 1))
        + 0.25 * clamp(delta / 0.20, 0, 1)
        + 0.20 * clamp((55 - distance) / 55, 0, 1)
    )
    if risk >= 0.45:
        return 5, "defensivo", "Riesgo alto: proteccion muy cercana para reducir cola y capital en riesgo."
    if risk >= 0.34:
        return 10, "prudente", "Riesgo medio: compra OTM cercana para no regalar demasiado riesgo por poca prima."
    if risk >= 0.23:
        return 20, "equilibrado", "Riesgo controlado: spread intermedio entre prima y proteccion."
    if risk >= 0.15:
        return 30, "ampliado", "Riesgo bajo: se amplia el spread para capturar mas prima."
    return 40, "expansivo", "Riesgo muy bajo: se permite una proteccion mas alejada para exprimir prima."


def choose_protection_option(candidate, options):
    if not options:
        return None
    target_width, profile, reason = protection_target(candidate)
    reasonable = [opt for opt in options if opt.get("reasonableCredit")]
    pool = reasonable or options

    def rank(option):
        width_gap = abs(option["width"] - target_width)
        credit = safe_num(option.get("creditUsd"), 0)
        rr = safe_num(option.get("riskRewardPct"), 0)
        if option["width"] < target_width and option.get("reasonableCredit"):
            width_gap -= 0.35
        return (width_gap, -min(credit, 220), -rr)

    selected = sorted(pool, key=rank)[0]
    selected = dict(selected)
    selected["targetWidth"] = target_width
    selected["profile"] = profile
    selected["reason"] = reason
    return selected


def apply_protection_profiles(candidates, context):
    chain = context["chain"]
    date_str = context["date"]
    for row in candidates:
        options = protection_options(chain, date_str, row["side"], row["strike"])
        selected = choose_protection_option(row, options)
        row["baseFinalScore"] = row.get("finalScore")
        row["baseWingCreditUsd"] = row.get("wingCreditUsd")
        row["protectionOptionsCount"] = len(options)
        if not selected:
            row["buyStrike"] = None
            row["spreadWidthPoints"] = None
            row["maxRiskUsd"] = None
            row["riskRewardPct"] = None
            row["protectionProfile"] = "sin_datos"
            row["protectionReason"] = "No hay datos suficientes para calcular la pata comprada."
            continue
        row["buyStrike"] = selected.get("buyStrike")
        row["spreadWidthPoints"] = selected.get("width")
        row["wingCreditUsd"] = selected.get("creditUsd")
        row["maxRiskUsd"] = selected.get("maxRiskUsd")
        row["riskRewardPct"] = selected.get("riskRewardPct")
        row["reasonableCredit"] = bool(selected.get("reasonableCredit"))
        row["isDry"] = bool(row.get("isDry") or not selected.get("reasonableCredit"))
        row["protectionProfile"] = selected.get("profile")
        row["protectionReason"] = selected.get("reason")
        credit = safe_num(row.get("wingCreditUsd"), 0)
        max_risk = safe_num(row.get("maxRiskUsd"), 10**9)
        spread_pct = safe_num(row.get("spreadPct"), 250)
        tradable = safe_num(row.get("tradableScore"), 0)
        premium_score = clamp((credit - MIN_REASONABLE_CREDIT_USD) / 85, 0, 1)
        liquidity_score = clamp(tradable / 20, 0, 1)
        risk_penalty = clamp(max_risk / 20000, 0, 0.16)
        spread_penalty = clamp(spread_pct / 140, 0, 0.42)
        dry_penalty = 0.24 if row.get("isDry") else 0
        row["finalScore"] = clamp(
            0.62 * safe_num(row.get("finalProbNoTouch"), 0.5)
            + 0.24 * premium_score
            + 0.10 * liquidity_score
            - risk_penalty
            - spread_penalty
            - dry_penalty,
            0,
            1,
        )
    return candidates


def collect_level_values(gex, dex, side):
    levels = set()
    consensus = (gex.get("consensus") or {}).get("next") or {}
    if side == "call":
        for key in ("callWallConsensus", "gammaFlipConsensus", "volTriggerConsensus"):
            levels.add(round_step(consensus.get(key)))
        levels.add(round_step((dex.get("levels") or {}).get("callDeltaWall")))
        levels.add(round_step((dex.get("levels") or {}).get("maxPositiveDex")))
    else:
        for key in ("putWallConsensus", "gammaFlipConsensus", "volTriggerConsensus"):
            levels.add(round_step(consensus.get(key)))
        levels.add(round_step((dex.get("levels") or {}).get("putDeltaWall")))
        levels.add(round_step((dex.get("levels") or {}).get("maxNegativeDex")))
    return {v for v in levels if v is not None}


def candidate_strikes(spotgamma, gex, dex, side):
    cw = safe_num(spotgamma.get("callWall"))
    pw = safe_num(spotgamma.get("putWall"))
    if side == "call":
        base = set(strike_range(cw - 100, cw + 25))
        base.add(round_step(cw))
    else:
        base = set(strike_range(pw - 25, pw + 100))
        base.add(round_step(pw))
    base |= collect_level_values(gex, dex, side)
    return sorted(s for s in base if s is not None)


def benchmark_from_signal(date_str):
    signal = load_json_safe(os.path.join(SIGNALS_DIR, f"{date_str}.json")) or {}
    legs = signal.get("legs") or {}
    open_wall = signal.get("openWall") or {}
    out = {
        "status": signal.get("status"),
        "entryCredit": safe_num(signal.get("entryCredit")),
        "entryCreditUsd": safe_num(signal.get("entryCredit")) * 100 if safe_num(signal.get("entryCredit")) is not None else None,
        "callStrike": None,
        "putStrike": None,
        "skippedWings": open_wall.get("skipped_wings") or [],
        "openPct": safe_num(open_wall.get("open_pct")),
        "adjustment": open_wall.get("adjustment"),
    }
    if legs.get("sell_call"):
        out["callStrike"] = safe_num(legs["sell_call"].get("strike"))
    if legs.get("sell_put"):
        out["putStrike"] = safe_num(legs["sell_put"].get("strike"))
    return out


def numeric_feature(row, key):
    value = row.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    value = safe_num(value)
    return value if value is not None else 0.0


def heuristic_no_touch_prob(candidate):
    distance = safe_num(candidate.get("distanceOpenPoints"), 0)
    spread_pct = safe_num(candidate.get("spreadPct"), 250)
    credit = safe_num(candidate.get("wingCreditUsd"), 0)
    delta = safe_num(candidate.get("absDelta"), 0.35)
    iv = safe_num(candidate.get("iv"), 0.1)
    gex_dist = safe_num(candidate.get("gexDistanceSideConsensus"), 0)
    dex_dist = safe_num(candidate.get("dexDistanceSideWall"), 0)
    range5 = safe_num(candidate.get("rangePct5dAvg"), 0.8)
    rv5 = safe_num(candidate.get("realizedVol5d"), 0.006)
    is_dry = 1 if candidate.get("isDry") else 0
    reasonable = 1 if candidate.get("reasonableCredit") else 0

    distance_score = clamp((distance - 20) / 140, -0.35, 1.15)
    delta_score = clamp((0.22 - delta) / 0.22, -0.5, 0.8)
    gex_buffer = clamp(gex_dist / 100, 0, 0.55)
    dex_buffer = clamp(dex_dist / 100, 0, 0.35)
    premium_tension = clamp((credit - 8) / 65, -0.15, 0.45)
    spread_penalty = clamp(spread_pct / 90, 0, 1.2)
    vol_penalty = clamp((range5 - 0.9) / 1.1, 0, 0.7) + clamp((rv5 - 0.006) / 0.01, 0, 0.55)
    iv_tension = clamp((iv - 0.16) / 0.20, 0, 0.45)
    raw = (
        -0.35
        + 2.15 * distance_score
        + 0.95 * delta_score
        + 0.55 * gex_buffer
        + 0.35 * dex_buffer
        + 0.28 * premium_tension
        + 0.22 * reasonable
        - 0.78 * spread_penalty
        - 0.68 * vol_penalty
        - 0.35 * iv_tension
        - 0.95 * is_dry
    )
    return clamp(logistic(raw), 0.02, 0.985)


class RegularizedLogisticRegression:
    def __init__(self, columns, l2=0.18, lr=0.045, epochs=340):
        self.columns = columns
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.means = {}
        self.stds = {}
        self.weights = [0.0] * (len(columns) + 1)
        self.active = False
        self.reason = "not_trained"

    def _matrix(self, rows, fit=False):
        if fit:
            for col in self.columns:
                values = [numeric_feature(row, col) for row in rows]
                avg = sum(values) / len(values) if values else 0.0
                sd = stdev(values) or 1.0
                self.means[col] = avg
                self.stds[col] = sd if sd > 1e-9 else 1.0
        matrix = []
        for row in rows:
            matrix.append([
                (numeric_feature(row, col) - self.means.get(col, 0.0)) / self.stds.get(col, 1.0)
                for col in self.columns
            ])
        return matrix

    def fit(self, rows):
        labels = [int(row.get("labelNoTouch")) for row in rows if row.get("labelNoTouch") is not None]
        if len(rows) < 60 or len(set(labels)) < 2:
            self.reason = "warmup_or_single_class"
            return self
        x = self._matrix(rows, fit=True)
        y = [int(row.get("labelNoTouch")) for row in rows]
        pos = sum(y)
        neg = len(y) - pos
        if not pos or not neg:
            self.reason = "single_class"
            return self
        class_weight = {1: len(y) / (2 * pos), 0: len(y) / (2 * neg)}
        self.weights = [0.0] * (len(self.columns) + 1)
        for _ in range(self.epochs):
            grads = [0.0] * len(self.weights)
            for xi, yi in zip(x, y):
                z = self.weights[0] + sum(w * v for w, v in zip(self.weights[1:], xi))
                p = logistic(z)
                err = (p - yi) * class_weight[yi]
                grads[0] += err
                for idx, value in enumerate(xi, start=1):
                    grads[idx] += err * value
            n = len(y)
            self.weights[0] -= self.lr * grads[0] / n
            for idx in range(1, len(self.weights)):
                reg = self.l2 * self.weights[idx]
                self.weights[idx] -= self.lr * (grads[idx] / n + reg)
        self.active = True
        self.reason = "trained"
        return self

    def predict_one(self, row):
        if not self.active:
            return 0.5
        xi = self._matrix([row], fit=False)[0]
        z = self.weights[0] + sum(w * v for w, v in zip(self.weights[1:], xi))
        return clamp(logistic(z), 0.02, 0.98)


class SmallGradientBoosting:
    def __init__(self, columns, estimators=14, learning_rate=0.075):
        self.columns = columns
        self.estimators = estimators
        self.learning_rate = learning_rate
        self.base_logit = 0.0
        self.stumps = []
        self.active = False
        self.reason = "not_trained"

    def fit(self, rows):
        labels = [int(row.get("labelNoTouch")) for row in rows if row.get("labelNoTouch") is not None]
        if len(rows) < 60 or len(set(labels)) < 2:
            self.reason = "warmup_or_single_class"
            return self
        y = [int(row.get("labelNoTouch")) for row in rows]
        pos_rate = clamp(sum(y) / len(y), 0.03, 0.97)
        self.base_logit = math.log(pos_rate / (1 - pos_rate))
        scores = [self.base_logit] * len(rows)
        min_leaf = max(8, int(len(rows) * 0.08))
        self.stumps = []
        for _ in range(self.estimators):
            residuals = [yi - logistic(si) for yi, si in zip(y, scores)]
            best = None
            for col in self.columns:
                pairs = sorted((numeric_feature(row, col), residuals[idx], idx) for idx, row in enumerate(rows))
                unique = sorted(set(v for v, _, _ in pairs))
                if len(unique) < 3:
                    continue
                step = max(1, len(unique) // 16)
                thresholds = [(unique[i - 1] + unique[i]) / 2 for i in range(1, len(unique), step)]
                for threshold in thresholds:
                    left = [r for v, r, _ in pairs if v <= threshold]
                    right = [r for v, r, _ in pairs if v > threshold]
                    if len(left) < min_leaf or len(right) < min_leaf:
                        continue
                    left_value = clamp(sum(left) / len(left), -0.8, 0.8)
                    right_value = clamp(sum(right) / len(right), -0.8, 0.8)
                    loss = 0.0
                    for v, r, _ in pairs:
                        pred = left_value if v <= threshold else right_value
                        loss += (r - pred) ** 2
                    if best is None or loss < best["loss"]:
                        best = {
                            "feature": col,
                            "threshold": threshold,
                            "left": left_value,
                            "right": right_value,
                            "loss": loss,
                        }
            if best is None:
                break
            self.stumps.append(best)
            for idx, row in enumerate(rows):
                value = numeric_feature(row, best["feature"])
                update = best["left"] if value <= best["threshold"] else best["right"]
                scores[idx] += self.learning_rate * update
        self.active = bool(self.stumps)
        self.reason = "trained" if self.active else "no_valid_stumps"
        return self

    def predict_one(self, row):
        if not self.active:
            return 0.5
        score = self.base_logit
        for stump in self.stumps:
            value = numeric_feature(row, stump["feature"])
            score += self.learning_rate * (stump["left"] if value <= stump["threshold"] else stump["right"])
        return clamp(logistic(score), 0.02, 0.98)


def build_candidate(date_str, side, strike, context):
    underlying = context["underlying"]
    prior_underlying = context.get("priorUnderlying") or underlying
    spotgamma = context["spotgamma"]
    gex = context["gex"]
    dex = context["dex"]
    vol_surface = context["vol_surface"]
    chain = context["chain"]
    benchmark = context["benchmark"]

    open_price = safe_num(spotgamma.get("open")) or safe_num(underlying.get("open")) or safe_num(chain.get("spot"))
    high = safe_num(underlying.get("high"))
    low = safe_num(underlying.get("low"))
    wall = safe_num(spotgamma.get("callWall" if side == "call" else "putWall"))
    distance_open = strike - open_price if side == "call" else open_price - strike
    if distance_open <= 0:
        return None

    wing = wing_snapshot(chain, date_str, side, strike)
    sell = wing.get("sell") or {}
    buy = wing.get("buy") or {}
    consensus = (gex.get("consensus") or {}).get("next") or {}
    dex_levels = dex.get("levels") or {}
    if side == "call":
        gex_side = safe_num(consensus.get("callWallConsensus"))
        gex_disp = safe_num(consensus.get("callWallDispersionRange"))
        gex_agree = safe_num(consensus.get("callWallAgreementWithin25Count"))
        dex_side = safe_num(dex_levels.get("callDeltaWall"))
        bench_strike = safe_num(benchmark.get("callStrike"))
        touched = high is not None and high >= strike
    else:
        gex_side = safe_num(consensus.get("putWallConsensus"))
        gex_disp = safe_num(consensus.get("putWallDispersionRange"))
        gex_agree = safe_num(consensus.get("putWallAgreementWithin25Count"))
        dex_side = safe_num(dex_levels.get("putDeltaWall"))
        bench_strike = safe_num(benchmark.get("putStrike"))
        touched = low is not None and low <= strike

    front = vol_surface.get("front") or {}
    term = vol_surface.get("termStructure") or {}
    quality = vol_surface.get("quality") or {}
    row = {
        "date": date_str,
        "side": side,
        "strike": strike,
        "side_call": 1 if side == "call" else 0,
        "entrySpot": safe_num(chain.get("spot")),
        "sessionOpen": open_price,
        "distanceOpenPoints": distance_open,
        "distance_open_points": distance_open,
        "distanceOpenPct": distance_open / open_price * 100 if open_price else None,
        "distance_open_pct": distance_open / open_price * 100 if open_price else None,
        "wall": wall,
        "distanceWallPoints": abs(strike - wall) if wall is not None else None,
        "distance_wall_points": abs(strike - wall) if wall is not None else None,
        "distanceBenchmarkPoints": abs(strike - bench_strike) if bench_strike is not None else None,
        "distance_benchmark_points": abs(strike - bench_strike) if bench_strike is not None else 999.0,
        "wall_range": safe_num(spotgamma.get("wallRange")),
        "open_pct_walls": safe_num(spotgamma.get("openPctWalls")),
        "wingCreditUsd": safe_num(wing.get("wingCreditUsd")),
        "wing_credit_usd": safe_num(wing.get("wingCreditUsd")),
        "reasonableCredit": wing.get("reasonableCredit"),
        "sellBidUsd": safe_num(sell.get("premiumUsd")),
        "sell_bid_usd": safe_num(sell.get("premiumUsd")),
        "sellMidUsd": safe_num(sell.get("midPremiumUsd")),
        "sell_mid_usd": safe_num(sell.get("midPremiumUsd")),
        "buyAskUsd": safe_num(buy.get("ask")) * 100 if safe_num(buy.get("ask")) is not None else None,
        "buy_ask_usd": safe_num(buy.get("ask")) * 100 if safe_num(buy.get("ask")) is not None else None,
        "spreadPct": safe_num(sell.get("spreadPct")),
        "spread_pct": safe_num(sell.get("spreadPct")),
        "iv": safe_num(sell.get("iv")),
        "absDelta": safe_num(sell.get("absDelta")),
        "abs_delta": safe_num(sell.get("absDelta")),
        "volume": safe_num(sell.get("volume")),
        "volume_log": log1p(sell.get("volume")),
        "openInterest": safe_num(sell.get("openInterest")),
        "oi_log": log1p(sell.get("openInterest")),
        "tradableScore": safe_num(sell.get("tradableScore")),
        "tradable_score": safe_num(sell.get("tradableScore")),
        "isDry": bool(sell.get("isDry") or not wing.get("reasonableCredit")),
        "is_dry": 1 if (sell.get("isDry") or not wing.get("reasonableCredit")) else 0,
        "gexDistanceSideConsensus": abs(strike - gex_side) if gex_side is not None else None,
        "gex_distance_side_consensus": abs(strike - gex_side) if gex_side is not None else 250.0,
        "gex_side_dispersion": gex_disp,
        "gex_side_agreement": gex_agree,
        "gex_net_positive_count": safe_num(consensus.get("positiveNetAtSpotCount")),
        "dexDistanceSideWall": abs(strike - dex_side) if dex_side is not None else None,
        "dex_distance_side_wall": abs(strike - dex_side) if dex_side is not None else 250.0,
        "dexDistanceFlip": abs(strike - safe_num(dex_levels.get("dexFlip"))) if safe_num(dex_levels.get("dexFlip")) is not None else None,
        "dex_distance_flip": abs(strike - safe_num(dex_levels.get("dexFlip"))) if safe_num(dex_levels.get("dexFlip")) is not None else 250.0,
        "dex_net_positive": 1 if dex_levels.get("netDexAtSpotSign") == "positive" else 0,
        "front_atm_iv": safe_num(front.get("atmIv")),
        "front_skew_balance": safe_num(front.get("skewBalance1Pct")),
        "front_minus_second": safe_num(term.get("frontMinusSecond")),
        "surface_missing_iv_pct": safe_num(quality.get("missingIvPct")),
        "vix_open": safe_num(underlying.get("vixOpen")),
        "gap_pct": safe_num(underlying.get("gapPct")),
        "rangePct5dAvg": safe_num(prior_underlying.get("rangePct5dAvg")),
        "range_pct_5d": safe_num(prior_underlying.get("rangePct5dAvg")),
        "realizedVol5d": safe_num(prior_underlying.get("realizedVol5d")),
        "realized_vol_5d": safe_num(prior_underlying.get("realizedVol5d")),
        "trend_5d": safe_num(prior_underlying.get("trend5d")),
        "labelTouched": touched if high is not None and low is not None else None,
        "labelNoTouch": (not touched) if high is not None and low is not None else None,
    }
    row["heuristicProbNoTouch"] = heuristic_no_touch_prob(row)
    return row


def build_candidates_for_date(date_str, context):
    candidates = []
    open_price = safe_num(context["spotgamma"].get("open")) or safe_num(context["underlying"].get("open")) or safe_num(context["chain"].get("spot"))
    for side in ("call", "put"):
        for strike in candidate_strikes(context["spotgamma"], context["gex"], context["dex"], side):
            if open_price is None:
                continue
            if side == "call" and strike <= open_price:
                continue
            if side == "put" and strike >= open_price:
                continue
            row = build_candidate(date_str, side, strike, context)
            if row:
                candidates.append(row)
    candidates.sort(key=lambda item: (item["side"], item["strike"]))
    return candidates


def fit_models(train_rows, train_day_count):
    usable = [row for row in train_rows if row.get("labelNoTouch") is not None]
    labels = [row.get("labelNoTouch") for row in usable]
    can_train = train_day_count >= MIN_TRAIN_DAYS and len(set(labels)) >= 2 and len(usable) >= 60
    lrr = RegularizedLogisticRegression(FEATURE_COLUMNS)
    gb = SmallGradientBoosting(FEATURE_COLUMNS)
    if can_train:
        lrr.fit(usable)
        gb.fit(usable)
    else:
        lrr.reason = "warmup_min_days"
        gb.reason = "warmup_min_days"
    return lrr, gb, usable


def score_candidates(candidates, train_rows, train_day_count):
    lrr, gb, usable = fit_models(train_rows, train_day_count)
    model_active = lrr.active and gb.active
    weights = {"heuristic": 0.70, "logistic": 0.15, "boosting": 0.15}
    if not model_active:
        weights = {"heuristic": 0.86, "logistic": 0.07, "boosting": 0.07}
    for row in candidates:
        logistic_prob = lrr.predict_one(row)
        boosting_prob = gb.predict_one(row)
        heuristic_prob = safe_num(row.get("heuristicProbNoTouch"), 0.5)
        no_touch_prob = (
            weights["heuristic"] * heuristic_prob
            + weights["logistic"] * logistic_prob
            + weights["boosting"] * boosting_prob
        )
        credit = safe_num(row.get("wingCreditUsd"), 0)
        spread_pct = safe_num(row.get("spreadPct"), 250)
        tradable = safe_num(row.get("tradableScore"), 0)
        premium_score = clamp((credit - MIN_REASONABLE_CREDIT_USD) / 65, 0, 1)
        liquidity_score = clamp(tradable / 20, 0, 1)
        spread_penalty = clamp(spread_pct / 120, 0, 0.45)
        dry_penalty = 0.22 if row.get("isDry") else 0
        final_score = clamp(
            0.64 * no_touch_prob + 0.24 * premium_score + 0.12 * liquidity_score - spread_penalty - dry_penalty,
            0,
            1,
        )
        row["logisticProbNoTouch"] = logistic_prob
        row["boostingProbNoTouch"] = boosting_prob
        row["finalProbNoTouch"] = no_touch_prob
        row["finalScore"] = final_score
    meta = {
        "modelStatus": "trained" if model_active else "warmup",
        "priorTrainingDays": train_day_count,
        "priorTrainingRows": len(usable),
        "weights": weights,
        "logisticStatus": "trained" if lrr.active else lrr.reason,
        "boostingStatus": "trained" if gb.active else gb.reason,
        "minTrainDays": MIN_TRAIN_DAYS,
        "featureCount": len(FEATURE_COLUMNS),
    }
    return candidates, meta


def choose_wings(candidates):
    calls = sorted([c for c in candidates if c["side"] == "call"], key=lambda r: r["finalScore"], reverse=True)
    puts = sorted([c for c in candidates if c["side"] == "put"], key=lambda r: r["finalScore"], reverse=True)

    def acceptable(row):
        return bool(
            row
            and row.get("reasonableCredit")
            and not row.get("isDry")
            and row.get("buyStrike") is not None
            and safe_num(row.get("finalProbNoTouch"), 0) >= 0.64
            and safe_num(row.get("finalScore"), 0) >= 0.48
        )

    selected = []
    if calls and acceptable(calls[0]):
        selected.append(calls[0])
    if puts and acceptable(puts[0]):
        selected.append(puts[0])
    if len(selected) == 2:
        action = "iron_condor"
    elif len(selected) == 1:
        action = "solo_call" if selected[0]["side"] == "call" else "solo_put"
    else:
        action = "no_operar"
    return action, selected, calls, puts


def slim_candidate(row):
    keys = [
        "side", "strike", "buyStrike", "spreadWidthPoints", "entrySpot", "sessionOpen", "distanceOpenPoints", "distanceOpenPct",
        "wingCreditUsd", "sellBidUsd", "sellMidUsd", "spreadPct", "iv", "absDelta",
        "volume", "openInterest", "tradableScore", "isDry", "reasonableCredit", "maxRiskUsd", "riskRewardPct",
        "protectionProfile", "protectionReason", "baseFinalScore", "baseWingCreditUsd",
        "heuristicProbNoTouch", "logisticProbNoTouch", "boostingProbNoTouch",
        "finalProbNoTouch", "finalScore", "labelTouched", "labelNoTouch",
        "gexDistanceSideConsensus", "dexDistanceSideWall",
    ]
    return {key: row.get(key) for key in keys}


def decision_commentary(action, selected, benchmark):
    if action == "no_operar" or not selected:
        return "El Shadow no encuentra una combinacion prima/riesgo suficientemente limpia para vender volatilidad."
    parts = []
    benchmark_credit = safe_num(benchmark.get("entryCreditUsd"))
    total_credit = sum(safe_num(row.get("wingCreditUsd"), 0) for row in selected)
    if benchmark_credit is not None:
        diff = total_credit - benchmark_credit
        parts.append(f"Credito Shadow ${round(total_credit)} frente a benchmark ${round(benchmark_credit)} ({diff:+.0f}).")
    else:
        parts.append(f"Credito Shadow estimado ${round(total_credit)}.")
    for row in selected:
        side = "call" if row.get("side") == "call" else "put"
        width = safe_num(row.get("spreadWidthPoints"))
        parts.append(
            f"Ala {side}: vende {int(row.get('strike'))}, compra {int(row.get('buyStrike')) if row.get('buyStrike') is not None else '-'} "
            f"({int(width) if width is not None else '-'} pts, perfil {row.get('protectionProfile') or '-'})."
        )
    return " ".join(parts)


def prediction_for_date(date_str, context, candidates, train_rows, train_day_count):
    scored, training = score_candidates(candidates, train_rows, train_day_count)
    scored = apply_protection_profiles(scored, context)
    action, selected, calls, puts = choose_wings(scored)
    selected_slim = [slim_candidate(row) for row in selected]
    labels_known = all(row.get("labelNoTouch") is not None for row in selected) if selected else False
    no_touch_all = all(row.get("labelNoTouch") is True for row in selected) if selected and labels_known else None
    total_credit = sum(safe_num(row.get("wingCreditUsd"), 0) for row in selected)
    total_max_risk = sum(safe_num(row.get("maxRiskUsd"), 0) for row in selected)
    avg_prob = mean([row.get("finalProbNoTouch") for row in selected])
    avg_score = mean([row.get("finalScore") for row in selected])
    benchmark = context["benchmark"]
    underlying = context["underlying"]
    spotgamma = context["spotgamma"]

    return {
        "version": VERSION,
        "date": date_str,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "mode": "paper_shadow",
        "objective": "Optimize no_touch with reasonable premium before using the model for live decisions.",
        "training": training,
        "context": {
            "entryChain": f"data/chains/{date_str}.json",
            "entrySpot": safe_num(context["chain"].get("spot")),
            "open": safe_num(spotgamma.get("open")) or safe_num(underlying.get("open")),
            "high": safe_num(underlying.get("high")),
            "low": safe_num(underlying.get("low")),
            "close": safe_num(underlying.get("close")),
            "callWall": safe_num(spotgamma.get("callWall")),
            "putWall": safe_num(spotgamma.get("putWall")),
            "openPctWalls": safe_num(spotgamma.get("openPctWalls")),
            "openZone": spotgamma.get("openZone"),
            "priorUnderlyingFeatureDate": context.get("priorUnderlyingDate"),
        },
        "benchmark": benchmark,
        "decision": {
            "action": action,
            "selectedWings": selected_slim,
            "totalCreditUsd": total_credit,
            "totalMaxRiskUsd": total_max_risk,
            "portfolioRiskRewardPct": total_credit / total_max_risk * 100 if total_max_risk else None,
            "averageProbNoTouch": avg_prob,
            "averageScore": avg_score,
            "confidence": "alta" if avg_prob and avg_prob >= 0.78 else "media" if avg_prob and avg_prob >= 0.68 else "baja",
            "commentary": decision_commentary(action, selected, benchmark),
            "note": "Shadow mode: no altera la senal real ni envia ordenes.",
        },
        "evaluation": {
            "available": labels_known,
            "noTouchAll": no_touch_all,
            "selectedTouchCount": sum(1 for row in selected if row.get("labelTouched") is True),
            "high": safe_num(underlying.get("high")),
            "low": safe_num(underlying.get("low")),
            "benchmarkNoTouch": benchmark_no_touch(benchmark, underlying),
        },
        "candidates": {
            "count": len(scored),
            "topCalls": [slim_candidate(row) for row in calls[:18]],
            "topPuts": [slim_candidate(row) for row in puts[:18]],
            "all": [slim_candidate(row) for row in sorted(scored, key=lambda r: r["finalScore"], reverse=True)],
        },
    }


def benchmark_no_touch(benchmark, underlying):
    high = safe_num(underlying.get("high"))
    low = safe_num(underlying.get("low"))
    if high is None or low is None:
        return None
    touched = []
    call_strike = safe_num(benchmark.get("callStrike"))
    put_strike = safe_num(benchmark.get("putStrike"))
    if call_strike is not None:
        touched.append(high >= call_strike)
    if put_strike is not None:
        touched.append(low <= put_strike)
    if not touched:
        return None
    return not any(touched)


def summary_from_predictions(predictions):
    evaluated = [p for p in predictions if p.get("evaluation", {}).get("available") is True and p.get("decision", {}).get("action") != "no_operar"]
    signals = [p for p in predictions if p.get("decision", {}).get("action") != "no_operar"]
    wins = [p for p in evaluated if p.get("evaluation", {}).get("noTouchAll") is True]
    bench_eval = [p for p in predictions if p.get("evaluation", {}).get("benchmarkNoTouch") is not None]
    bench_wins = [p for p in bench_eval if p.get("evaluation", {}).get("benchmarkNoTouch") is True]
    credits = [safe_num(p.get("decision", {}).get("totalCreditUsd")) for p in signals]
    credits = [c for c in credits if c is not None]
    max_risks = [safe_num(p.get("decision", {}).get("totalMaxRiskUsd")) for p in signals]
    max_risks = [r for r in max_risks if r is not None and r > 0]
    rr = [safe_num(p.get("decision", {}).get("portfolioRiskRewardPct")) for p in signals]
    rr = [r for r in rr if r is not None]
    return {
        "predictions": len(predictions),
        "shadowSignals": len(signals),
        "shadowNoTrade": len(predictions) - len(signals),
        "evaluatedShadowSignals": len(evaluated),
        "shadowNoTouchWins": len(wins),
        "shadowNoTouchWr": len(wins) / len(evaluated) * 100 if evaluated else None,
        "benchmarkEvaluatedSignals": len(bench_eval),
        "benchmarkNoTouchWr": len(bench_wins) / len(bench_eval) * 100 if bench_eval else None,
        "averageShadowCreditUsd": mean(credits),
        "averageShadowMaxRiskUsd": mean(max_risks),
        "averageShadowRiskRewardPct": mean(rr),
    }


def build():
    os.makedirs(OUT_PREDICTIONS_DIR, exist_ok=True)
    underlying = load_rows("subyacente", key="date")
    spotgamma = load_rows("spotgamma", key="date")
    gex = load_rows("gex", key="targetSession")
    dex = load_rows("dex", key="targetSession")
    vol_surface = load_rows("vol_surface", key="targetSession")
    chains_index = load_json_safe(os.path.join(DATA_DIR, "chains-index.json")) or {"dates": []}
    dates = sorted(
        set(chains_index.get("dates") or [])
        & set(underlying.keys())
        & set(spotgamma.keys())
        & set(gex.keys())
        & set(dex.keys())
        & set(vol_surface.keys())
    )

    underlying_dates = sorted(underlying.keys())
    prior_underlying_by_date = {}
    previous = None
    for date_str in underlying_dates:
        prior_underlying_by_date[date_str] = previous
        previous = underlying[date_str]

    candidate_history = {}
    predictions = []
    training_rows = []
    training_dates = set()
    for date_str in dates:
        chain = load_json_safe(os.path.join(CHAINS_DIR, f"{date_str}.json")) or {}
        if not chain.get("expirations"):
            continue
        context = {
            "date": date_str,
            "underlying": underlying[date_str],
            "priorUnderlying": prior_underlying_by_date.get(date_str) or underlying[date_str],
            "priorUnderlyingDate": (prior_underlying_by_date.get(date_str) or {}).get("date"),
            "spotgamma": spotgamma[date_str],
            "gex": gex[date_str],
            "dex": dex[date_str],
            "vol_surface": vol_surface[date_str],
            "chain": chain,
            "benchmark": benchmark_from_signal(date_str),
        }
        candidates = build_candidates_for_date(date_str, context)
        if not candidates:
            continue
        prediction = prediction_for_date(date_str, context, candidates, training_rows, len(training_dates))
        predictions.append(prediction)
        candidate_history[date_str] = candidates
        save_json(os.path.join(OUT_PREDICTIONS_DIR, f"{date_str}.json"), prediction)
        labeled = [row for row in candidates if row.get("labelNoTouch") is not None]
        if labeled:
            training_rows.extend(labeled)
            training_dates.add(date_str)

    latest = predictions[-1] if predictions else None
    now = datetime.now(timezone.utc).isoformat()
    index = {
        "version": VERSION,
        "lastUpdated": now,
        "mode": "paper_shadow",
        "description": "ML Shadow V1: candidate generator + heuristic + regularized logistic regression + small gradient boosting. No production signals are changed.",
        "objective": "Improve operate/no-operate and wing/strike selection for 0DTE volatility selling.",
        "dependencies": "pure_python_no_external_ml_libraries",
        "files": {
            "predictionsDir": "data/ml-shadow/predictions",
        },
        "model": {
            "candidateStepPoints": STRIKE_STEP,
            "baseScoringSpreadWidthPoints": SPREAD_WIDTH,
            "dynamicProtectionWidths": list(PROTECTION_WIDTHS),
            "minReasonableCreditUsd": MIN_REASONABLE_CREDIT_USD,
            "minTrainDays": MIN_TRAIN_DAYS,
            "featureColumns": FEATURE_COLUMNS,
        },
        "dates": [p["date"] for p in predictions],
        "latestDate": latest.get("date") if latest else None,
        "latestFile": f"data/ml-shadow/predictions/{latest.get('date')}.json" if latest else None,
        "summary": summary_from_predictions(predictions),
    }
    save_json(OUT_INDEX, index)
    print(f"OK ML Shadow: {len(predictions)} predictions, latest={index['latestDate']}")
    return index


def main():
    build()


if __name__ == "__main__":
    main()
