#!/usr/bin/env python3
"""Capture intraday 0DTE chains and premium evolution for SPX strategies."""
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fetch_daily import (
    CLOSE_CHAINS_INDEX,
    DATA_DIR,
    NYSE_HOLIDAYS,
    http_get_json,
    is_trading_day,
    load_json_safe,
    merge_strikes,
    parse_option,
    save_json,
)

HISTORY_DIR = os.path.join(DATA_DIR, "premium-history")
HISTORY_INDEX = os.path.join(DATA_DIR, "premium-history-index.json")
SHADOW_HISTORY_DIR = os.path.join(DATA_DIR, "premium-history-ml-shadow")
SHADOW_HISTORY_INDEX = os.path.join(DATA_DIR, "premium-history-ml-shadow-index.json")
INTRADAY_CHAINS_DIR = os.path.join(DATA_DIR, "chains-intraday")
INTRADAY_CHAINS_INDEX = os.path.join(DATA_DIR, "chains-intraday-index.json")
OHLC_FILE = os.path.join(DATA_DIR, "daily-ohlc.json")
CLOSE_DIR = os.path.join(DATA_DIR, "chains-close")
ENTRY_DIR = os.path.join(DATA_DIR, "chains")
ML_SHADOW_DIR = os.path.join(DATA_DIR, "ml-shadow", "predictions")
SPOTGAMMA_FILE = os.path.join(DATA_DIR, "spotgamma-levels.json")
GEX_MODELS_DIR = os.path.join(DATA_DIR, "gex-models")
GEX_MODELS_INDEX = os.path.join(DATA_DIR, "gex-models-index.json")
CONTRACT_MULTIPLIER = 100
SPREAD_WIDTH = 20
MIN_WING_CREDIT_USD = 5
MIN_WING_CREDIT_POINTS = MIN_WING_CREDIT_USD / CONTRACT_MULTIPLIER
MIN_T = 1 / 252
RISK_FREE_RATE = 0.04
CHAIN_MAX_PCT = 8.0
MIN_CAPTURE_INTERVAL_MINUTES = 4
ENTRY_FALLBACK_AFTER = (16, 7)


def normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_gamma(spot, strike, sigma, years_to_expiry):
    if not (spot > 0 and strike > 0):
        return 0
    vol = min(max(sigma if sigma > 0 else 0.18, 0.03), 5)
    t = max(years_to_expiry, MIN_T)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    return normal_pdf(d1) / (spot * vol * sqrt_t)


def exposure(spot, strike, sigma, years_to_expiry, oi, sign):
    return sign * bs_gamma(spot, strike, sigma, years_to_expiry) * max(oi or 0, 0) * 100 * spot * spot * 0.01


def nearest_expiration(chain):
    expirations = chain.get("expirations", {})
    if not expirations:
        return None, None
    return min(expirations.items(), key=lambda item: abs(float(item[1].get("dte") or 0)))


def compute_levels(chain):
    spot = float(chain.get("spot") or 0)
    _, expiration = nearest_expiration(chain)
    if not (spot > 0 and expiration):
        return None
    t = max(float(expiration.get("dte") or 0) / 252, MIN_T)
    rows = []
    for raw in expiration.get("strikes", []):
        strike = float(raw.get("strike") or 0)
        if strike <= 0:
            continue
        call_iv = float(raw.get("call_iv") or raw.get("put_iv") or 0.18)
        put_iv = float(raw.get("put_iv") or raw.get("call_iv") or 0.18)
        call_gex = exposure(spot, strike, call_iv, t, float(raw.get("call_oi") or 0), 1)
        put_gex = exposure(spot, strike, put_iv, t, float(raw.get("put_oi") or 0), -1)
        rows.append({"strike": strike, "call_gex": call_gex, "put_gex": put_gex})
    if not rows:
        return None
    call_wall = max(rows, key=lambda row: row["call_gex"])["strike"]
    put_wall = max(rows, key=lambda row: abs(row["put_gex"]))["strike"]
    return {"call_wall": call_wall, "put_wall": put_wall}


def load_spotgamma_levels(day):
    data = load_json_safe(SPOTGAMMA_FILE) or {}
    row = (data.get("byDate") or {}).get(day)
    if not row:
        return None
    call_wall = row.get("callWall")
    put_wall = row.get("putWall")
    if call_wall is None or put_wall is None:
        return None
    return {
        "call_wall": float(call_wall),
        "put_wall": float(put_wall),
        "vol_trigger": row.get("volTrigger"),
        "gamma_flip": row.get("gammaFlip"),
        "source": row.get("source") or "SpotGamma manual",
        "date": day,
        "sourceCloseDate": day,
        "levelsOrigin": "spotgamma",
    }


def level_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def round_level(value, step=25):
    if value is None:
        return None
    return float(round(float(value) / step) * step)


def level_component(component_id, name, levels, source_date=None, origin=None):
    return {
        "id": component_id,
        "name": name,
        "origin": origin or component_id,
        "sourceDate": source_date,
        "call_wall": level_float(levels.get("call_wall") if "call_wall" in levels else levels.get("callWall")),
        "put_wall": level_float(levels.get("put_wall") if "put_wall" in levels else levels.get("putWall")),
        "vol_trigger": level_float(levels.get("vol_trigger") if "vol_trigger" in levels else levels.get("volTrigger")),
        "gamma_flip": level_float(levels.get("gamma_flip") if "gamma_flip" in levels else levels.get("gammaFlip")),
    }


def combine_component_level(components, key):
    values = [component.get(key) for component in components if component.get(key) is not None]
    if not values:
        return None
    return round_level(sum(values) / len(values))


def load_gex_consensus_levels(day):
    index = load_json_safe(GEX_MODELS_INDEX) or {"dates": []}
    for source_date in sorted(index.get("dates", []), reverse=True):
        record = load_json_safe(os.path.join(GEX_MODELS_DIR, f"{source_date}.json"))
        if not record or record.get("targetSession") != day:
            continue
        models = record.get("models") or {}
        components = []
        for model_id in ("pure_gex", "liquidity_weighted"):
            model = models.get(model_id) or {}
            levels = model.get("next") or {}
            component = level_component(model_id, model.get("name") or model_id, levels, source_date, "gex_model")
            if component["call_wall"] is not None and component["put_wall"] is not None:
                components.append(component)
        if not components:
            return None
        spotgamma = load_spotgamma_levels(day)
        if spotgamma:
            components.append(level_component("spotgamma", spotgamma.get("source") or "SpotGamma manual", spotgamma, day, "spotgamma"))
        call_wall = combine_component_level(components, "call_wall")
        put_wall = combine_component_level(components, "put_wall")
        if call_wall is None or put_wall is None:
            return None
        vol_trigger = combine_component_level(components, "vol_trigger")
        gamma_flip = combine_component_level(components, "gamma_flip")
        component_ids = [component["id"] for component in components]
        return {
            "call_wall": call_wall,
            "put_wall": put_wall,
            "vol_trigger": vol_trigger,
            "gamma_flip": gamma_flip,
            "source": f"GEX consensus - {' + '.join(component_ids)} - close {source_date}",
            "date": day,
            "sourceCloseDate": source_date,
            "levelsOrigin": "gex_consensus",
            "levelsModel": "+".join(component_ids),
            "levelsComponents": components,
            "spotgammaAvailable": bool(spotgamma),
        }
    return None


def load_trading_levels(day):
    return load_gex_consensus_levels(day)


def previous_close_chain(today):
    index = load_json_safe(CLOSE_CHAINS_INDEX) or {"dates": []}
    previous = sorted((date for date in index.get("dates", []) if date < today), reverse=True)
    if not previous:
        return None, None
    date = previous[0]
    return date, load_json_safe(os.path.join(CLOSE_DIR, f"{date}.json"))


def nearest_strike(strikes, target):
    return min(strikes, key=lambda row: abs(float(row.get("strike") or 0) - target), default=None)


def q(row, key):
    value = row.get(key) if row else None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else 0
    except (TypeError, ValueError):
        return 0


def is_opex_day(date_string):
    dt = datetime.strptime(date_string, "%Y-%m-%d").date()
    if dt.month not in (3, 6, 9, 12) or dt.weekday() != 4:
        return False
    return 15 <= dt.day <= 21


def get_open_wall_setup(today, levels):
    call_wall = float(levels["call_wall"])
    put_wall = float(levels["put_wall"])
    wall_range = call_wall - put_wall
    if wall_range <= 0:
        return {
            "ok": False,
            "reason": "Rango de walls no válido",
            "call_wall": call_wall,
            "put_wall": put_wall,
        }
    ohlc = (load_json_safe(OHLC_FILE) or {}).get("byDate", {})
    today_open = ohlc.get(today, {}).get("spx_open")
    if today_open is None:
        return {
            "ok": False,
            "reason": "Open no disponible",
            "call_wall": call_wall,
            "put_wall": put_wall,
        }
    open_value = float(today_open)
    open_pct = (open_value - put_wall) / wall_range * 100
    sell_call = call_wall
    sell_put = put_wall
    adjustment = "Sin ajuste"
    if open_pct < 10 or open_pct > 90:
        return {
            "ok": False,
            "reason": f"Open fuera del rango operable ({open_pct:.2f}%)",
            "open": open_value,
            "open_pct": open_pct,
            "call_wall": call_wall,
            "put_wall": put_wall,
        }
    if 10 <= open_pct <= 20:
        sell_put = put_wall - 35
        adjustment = "Put Wall -35"
    elif 20 < open_pct <= 30:
        sell_put = put_wall - 20
        adjustment = "Put Wall -20"
    elif 80 < open_pct <= 90:
        sell_call = call_wall + 15
        adjustment = "Call Wall +15"
    return {
        "ok": True,
        "reason": None,
        "open": open_value,
        "open_pct": open_pct,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "sell_call": sell_call,
        "sell_put": sell_put,
        "adjustment": adjustment,
    }


def no_trade_reason(today, levels):
    setup = get_open_wall_setup(today, levels)
    return None if setup["ok"] else setup["reason"]


def wing_credit(legs, sell_name, buy_name):
    sell = legs.get(sell_name)
    buy = legs.get(buy_name)
    if not sell or not buy:
        return None
    return float(sell["entryBid"]) - float(buy["entryAsk"])


def remove_wing_if_unprofitable(legs, setup, wing, sell_name, buy_name):
    credit = wing_credit(legs, sell_name, buy_name)
    if credit is None or credit > MIN_WING_CREDIT_POINTS:
        return
    sell = legs.pop(sell_name, None)
    buy = legs.pop(buy_name, None)
    setup.setdefault("skipped_wings", []).append({
        "wing": wing,
        "reason": "NO COMPENSA",
        "credit": credit,
        "creditUsd": credit * CONTRACT_MULTIPLIER,
        "thresholdUsd": MIN_WING_CREDIT_USD,
        "sellStrike": sell.get("strike") if sell else None,
        "buyStrike": buy.get("strike") if buy else None,
    })


def no_compensa_reason(setup):
    skipped = setup.get("skipped_wings") or []
    if not skipped:
        return None
    labels = []
    for item in skipped:
        wing = "CALL" if item.get("wing") == "call" else "PUT"
        credit = item.get("creditUsd")
        credit_text = f"{credit:.2f} USD" if isinstance(credit, (int, float)) else "n/d"
        labels.append(f"{wing}: NO COMPENSA ({credit_text})")
    return "; ".join(labels)


def select_legs(today, levels):
    entry = load_json_safe(os.path.join(ENTRY_DIR, f"{today}.json"))
    if not entry:
        entry = load_intraday_entry_chain(today)
    if not entry:
        raise RuntimeError("la cadena de entrada todavía no existe")
    _, expiration = nearest_expiration(entry)
    strikes = expiration.get("strikes", []) if expiration else []
    setup = get_open_wall_setup(today, levels)
    if not setup["ok"]:
        raise RuntimeError(setup["reason"])
    targets = {
        "sell_call": setup["sell_call"],
        "buy_call": setup["sell_call"] + SPREAD_WIDTH,
        "sell_put": setup["sell_put"],
        "buy_put": setup["sell_put"] - SPREAD_WIDTH,
    }
    selected = {name: nearest_strike(strikes, target) for name, target in targets.items()}
    if any(row is None for row in selected.values()):
        raise RuntimeError("no se encontraron los cuatro strikes")
    legs = {}
    for name, row in selected.items():
        option_type = "call" if "call" in name else "put"
        legs[name] = {
            "type": option_type,
            "side": "sell" if name.startswith("sell") else "buy",
            "strike": float(row["strike"]),
            "entryBid": q(row, f"{option_type}_bid"),
            "entryAsk": q(row, f"{option_type}_ask"),
        }
    if is_opex_day(today):
        legs.pop("sell_put", None)
        legs.pop("buy_put", None)
    remove_wing_if_unprofitable(legs, setup, "call", "sell_call", "buy_call")
    remove_wing_if_unprofitable(legs, setup, "put", "sell_put", "buy_put")
    entry_credit = sum(
        leg["entryBid"] if leg["side"] == "sell" else -leg["entryAsk"]
        for leg in legs.values()
    )
    return entry, legs, entry_credit, setup


def fetch_current_chain(today):
    raw = http_get_json("https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json")
    data = raw.get("data", {})
    spot = data.get("current_price") or data.get("last") or data.get("price")
    parsed = [parse_option(option) for option in data.get("options", [])]
    parsed = [option for option in parsed if option and option["expiration"] == today]
    return float(spot or 0), merge_strikes(parsed, today, float(spot or 0), max_pct=CHAIN_MAX_PCT)


def make_snapshot(now_iso, spot, strikes, legs, entry_credit):
    quotes = {}
    close_cost = 0
    for name, leg in legs.items():
        row = nearest_strike(strikes, leg["strike"])
        prefix = leg["type"]
        bid = q(row, f"{prefix}_bid")
        ask = q(row, f"{prefix}_ask")
        quotes[name] = {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}
        close_cost += ask if leg["side"] == "sell" else -bid
    effective = datetime.fromisoformat(now_iso).astimezone(timezone.utc) - timedelta(minutes=15)
    return {
        "capturedAt": now_iso,
        "effectiveAt": effective.isoformat(),
        "spot": spot,
        "quotes": quotes,
        "closeCost": close_cost,
        "pnl": (entry_credit - close_cost) * CONTRACT_MULTIPLIER,
        "multiple": close_cost / entry_credit if entry_credit > 0 else None,
    }


def effective_at(now_iso):
    return (datetime.fromisoformat(now_iso).astimezone(timezone.utc) - timedelta(minutes=15)).isoformat()


def parse_iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def is_after_shadow_entry_time(now_madrid):
    return (now_madrid.hour, now_madrid.minute) >= ENTRY_FALLBACK_AFTER


def capture_after_entry_time(capture):
    captured = parse_iso(capture.get("capturedAt"))
    if not captured:
        return False
    madrid = captured.astimezone(ZoneInfo("Europe/Madrid"))
    return is_after_shadow_entry_time(madrid)


def load_data_file(rel_file):
    if not rel_file:
        return None
    rel_file = str(rel_file).replace("\\", "/")
    if rel_file.startswith("data/"):
        rel_file = rel_file[len("data/"):]
    return load_json_safe(os.path.join(DATA_DIR, *rel_file.split("/")))


def load_intraday_entry_chain(today):
    index = load_json_safe(INTRADAY_CHAINS_INDEX) or {}
    captures = ((index.get("byDate") or {}).get(today) or [])
    captures = sorted(captures, key=lambda item: item.get("capturedAt") or item.get("file") or "")
    fallback_pool = [capture for capture in captures if capture_after_entry_time(capture)] or captures
    for capture in fallback_pool:
        chain = load_data_file(capture.get("file"))
        if chain and chain.get("expirations"):
            return chain
    return None


def load_shadow_entry_chain(today, prediction=None):
    prediction = prediction or {}
    rel_file = ((prediction.get("context") or {}).get("entryChain"))
    chain = load_data_file(rel_file)
    if chain and chain.get("expirations"):
        return chain
    chain = load_json_safe(os.path.join(ENTRY_DIR, f"{today}.json"))
    if chain and chain.get("expirations"):
        return chain
    return load_intraday_entry_chain(today)


def recent_capture_exists(today, now_iso):
    index = load_json_safe(INTRADAY_CHAINS_INDEX) or {}
    captures = ((index.get("byDate") or {}).get(today) or [])
    if not captures:
        return False
    last = parse_iso(captures[-1].get("capturedAt"))
    now = parse_iso(now_iso)
    if not last or not now:
        return False
    return (now - last).total_seconds() < MIN_CAPTURE_INTERVAL_MINUTES * 60


def update_index_file(index_file, today, now_iso):
    index = load_json_safe(index_file) or {"dates": []}
    if today not in index["dates"]:
        index["dates"].append(today)
        index["dates"].sort()
    index["lastUpdated"] = now_iso
    save_json(index_file, index)


def save_intraday_chain(today, now_iso, spot, strikes):
    folder = os.path.join(INTRADAY_CHAINS_DIR, today)
    filename = datetime.fromisoformat(now_iso).strftime("%H%M%S") + "Z.json"
    rel_file = f"data/chains-intraday/{today}/{filename}"
    chain_file = os.path.join(folder, filename)
    payload = {
        "date": today,
        "kind": "intraday_0dte",
        "capturedAt": now_iso,
        "effectiveAt": effective_at(now_iso),
        "spot": spot,
        "source": "CBOE delayed quotes",
        "delayMinutesApprox": 15,
        "maxPctAroundSpot": CHAIN_MAX_PCT,
        "expirations": {
            today: {
                "dte": 0,
                "strikes": strikes,
            }
        },
    }
    save_json(chain_file, payload)

    index = load_json_safe(INTRADAY_CHAINS_INDEX) or {"dates": [], "byDate": {}}
    if today not in index["dates"]:
        index["dates"].append(today)
        index["dates"].sort()
    captures = index.setdefault("byDate", {}).setdefault(today, [])
    captures.append({
        "capturedAt": now_iso,
        "effectiveAt": payload["effectiveAt"],
        "spot": spot,
        "strikeCount": len(strikes),
        "file": rel_file,
    })
    captures.sort(key=lambda row: row.get("capturedAt") or "")
    index["lastUpdated"] = now_iso
    save_json(INTRADAY_CHAINS_INDEX, index)
    return rel_file


def ensure_benchmark_history(today, levels):
    history_file = os.path.join(HISTORY_DIR, f"{today}.json")
    history = load_json_safe(history_file)
    if history:
        return history, history_file
    reason = no_trade_reason(today, levels)
    if reason:
        history = {
            "date": today,
            "status": "no_trade",
            "reason": reason,
            "sourceCloseDate": levels.get("sourceCloseDate") or today,
            "levelsSource": levels.get("source"),
            "levelsOrigin": levels.get("levelsOrigin"),
            "levelsModel": levels.get("levelsModel"),
            "levelsComponents": levels.get("levelsComponents"),
            "spotgammaAvailable": levels.get("spotgammaAvailable"),
            "openWall": get_open_wall_setup(today, levels),
            "snapshots": [],
        }
        save_json(history_file, history)
        return history, history_file
    entry, legs, entry_credit, setup = select_legs(today, levels)
    if not legs:
        reason = no_compensa_reason(setup) or "NO COMPENSA"
        history = {
            "date": today,
            "status": "no_trade",
            "reason": reason,
            "sourceCloseDate": levels.get("sourceCloseDate") or today,
            "levelsSource": levels.get("source"),
            "levelsOrigin": levels.get("levelsOrigin"),
            "levelsModel": levels.get("levelsModel"),
            "levelsComponents": levels.get("levelsComponents"),
            "spotgammaAvailable": levels.get("spotgammaAvailable"),
            "openWall": setup,
            "snapshots": [],
        }
        save_json(history_file, history)
        return history, history_file
    history = {
        "date": today,
        "status": "active",
        "sourceCloseDate": levels.get("sourceCloseDate") or today,
        "levelsSource": levels.get("source"),
        "levelsOrigin": levels.get("levelsOrigin"),
        "levelsModel": levels.get("levelsModel"),
        "levelsComponents": levels.get("levelsComponents"),
        "spotgammaAvailable": levels.get("spotgammaAvailable"),
        "selectedAt": entry.get("capturedAt"),
        "entrySpot": entry.get("spot"),
        "entryCredit": entry_credit,
        "spreadWidth": SPREAD_WIDTH,
        "openWall": setup,
        "legs": legs,
        "snapshots": [],
    }
    save_json(history_file, history)
    return history, history_file


def entry_quote_for_leg(entry_strikes, leg):
    row = nearest_strike(entry_strikes, float(leg["strike"]))
    prefix = leg["type"]
    leg["entryBid"] = q(row, f"{prefix}_bid")
    leg["entryAsk"] = q(row, f"{prefix}_ask")
    return leg


def shadow_legs_from_prediction(today, prediction):
    entry = load_shadow_entry_chain(today, prediction)
    _, expiration = nearest_expiration(entry or {})
    entry_strikes = expiration.get("strikes", []) if expiration else []
    legs = {}
    for wing in (prediction.get("decision") or {}).get("selectedWings") or []:
        side = wing.get("side")
        sell_strike = wing.get("strike")
        buy_strike = wing.get("buyStrike")
        if side not in ("call", "put") or sell_strike is None or buy_strike is None:
            continue
        legs[f"sell_{side}"] = entry_quote_for_leg(entry_strikes, {
            "type": side,
            "side": "sell",
            "strike": float(sell_strike),
        })
        legs[f"buy_{side}"] = entry_quote_for_leg(entry_strikes, {
            "type": side,
            "side": "buy",
            "strike": float(buy_strike),
        })
    return entry, legs


def ensure_shadow_prediction(today, now_madrid):
    prediction_file = os.path.join(ML_SHADOW_DIR, f"{today}.json")
    if load_json_safe(prediction_file):
        return True
    if not is_after_shadow_entry_time(now_madrid):
        return False
    print("  INFO ML Shadow: no hay prediccion; se intenta generar con la cadena intradia disponible.")
    try:
        import ml_shadow

        ml_shadow.build()
    except Exception as exc:
        print(f"  WARN ML Shadow: no se pudo generar prediccion fallback ({exc}).")
    return bool(load_json_safe(prediction_file))


def ensure_shadow_history(today):
    history_file = os.path.join(SHADOW_HISTORY_DIR, f"{today}.json")
    history = load_json_safe(history_file)
    if history:
        return history, history_file
    prediction = load_json_safe(os.path.join(ML_SHADOW_DIR, f"{today}.json"))
    if not prediction:
        return None, history_file
    decision = prediction.get("decision") or {}
    if decision.get("action") == "no_operar":
        history = {
            "date": today,
            "status": "no_trade",
            "reason": decision.get("commentary") or "ML Shadow no propone operacion.",
            "modelVersion": prediction.get("version"),
            "generatedAt": prediction.get("generatedAt"),
            "snapshots": [],
        }
        save_json(history_file, history)
        return history, history_file
    entry, legs = shadow_legs_from_prediction(today, prediction)
    if not legs:
        return None, history_file
    entry_credit = decision.get("totalCreditUsd")
    entry_credit = float(entry_credit) / CONTRACT_MULTIPLIER if entry_credit is not None else sum(
        leg["entryBid"] if leg["side"] == "sell" else -leg["entryAsk"]
        for leg in legs.values()
    )
    history = {
        "date": today,
        "status": "active",
        "strategy": "ml_shadow",
        "modelVersion": prediction.get("version"),
        "generatedAt": prediction.get("generatedAt"),
        "selectedAt": entry.get("capturedAt") if entry else None,
        "entrySpot": entry.get("spot") if entry else None,
        "entryCredit": entry_credit,
        "entryCreditUsd": entry_credit * CONTRACT_MULTIPLIER,
        "action": decision.get("action"),
        "confidence": decision.get("confidence"),
        "legs": legs,
        "snapshots": [],
    }
    save_json(history_file, history)
    return history, history_file


def append_strategy_snapshot(history, history_file, now_iso, spot, strikes, index_file):
    if not history or history.get("status") != "active":
        return None
    snapshot = make_snapshot(now_iso, spot, strikes, history["legs"], history["entryCredit"])
    snapshots = history.setdefault("snapshots", [])
    if not any(item.get("capturedAt") == now_iso for item in snapshots):
        snapshots.append(snapshot)
        snapshots.sort(key=lambda item: item.get("capturedAt") or "")
    history["lastUpdated"] = now_iso
    save_json(history_file, history)
    update_index_file(index_file, history["date"], now_iso)
    return snapshot


def legacy_main_selected_premiums_only():
    now_madrid = datetime.now(ZoneInfo("Europe/Madrid"))
    today = now_madrid.date().isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()
    print(f"[{now_iso}] Premium monitor (Madrid {now_madrid:%H:%M})")
    if not is_trading_day(now_madrid.date()):
        print("  SKIP: no es día de mercado.")
        return
    if not (16 <= now_madrid.hour < 23):
        print("  SKIP: fuera de la ventana 16:00-22:59 Madrid.")
        return

    history_file = os.path.join(HISTORY_DIR, f"{today}.json")
    history = load_json_safe(history_file)
    if not history:
        levels = load_trading_levels(today)
        if not levels:
            raise RuntimeError(f"no hay niveles operativos cargados para {today}")
        reason = no_trade_reason(today, levels)
        if reason:
            history = {
                "date": today,
                "status": "no_trade",
                "reason": reason,
                "sourceCloseDate": levels.get("sourceCloseDate") or today,
                "levelsSource": levels.get("source"),
                "levelsOrigin": levels.get("levelsOrigin"),
                "levelsModel": levels.get("levelsModel"),
                "levelsComponents": levels.get("levelsComponents"),
                "spotgammaAvailable": levels.get("spotgammaAvailable"),
                "openWall": get_open_wall_setup(today, levels),
                "snapshots": [],
            }
            save_json(history_file, history)
            print(f"  SKIP: {reason}")
            update_index(today, now_iso)
            return
        try:
            entry, legs, entry_credit, setup = select_legs(today, levels)
        except RuntimeError as exc:
            if "cadena de entrada" in str(exc):
                print(f"  SKIP: {exc}. Se reintentará en la siguiente captura.")
                return
            raise
        if not legs:
            reason = no_compensa_reason(setup) or "NO COMPENSA"
            history = {
                "date": today,
                "status": "no_trade",
                "reason": reason,
                "sourceCloseDate": levels.get("sourceCloseDate") or today,
                "levelsSource": levels.get("source"),
                "levelsOrigin": levels.get("levelsOrigin"),
                "levelsModel": levels.get("levelsModel"),
                "levelsComponents": levels.get("levelsComponents"),
                "spotgammaAvailable": levels.get("spotgammaAvailable"),
                "openWall": setup,
                "snapshots": [],
            }
            save_json(history_file, history)
            print(f"  SKIP: {reason}")
            update_index(today, now_iso)
            return
        history = {
            "date": today,
            "status": "active",
            "sourceCloseDate": levels.get("sourceCloseDate") or today,
            "levelsSource": levels.get("source"),
            "levelsOrigin": levels.get("levelsOrigin"),
            "levelsModel": levels.get("levelsModel"),
            "levelsComponents": levels.get("levelsComponents"),
            "spotgammaAvailable": levels.get("spotgammaAvailable"),
            "selectedAt": entry.get("capturedAt"),
            "entrySpot": entry.get("spot"),
            "entryCredit": entry_credit,
            "spreadWidth": SPREAD_WIDTH,
            "openWall": setup,
            "legs": legs,
            "snapshots": [],
        }

    if history.get("status") != "active":
        print(f"  SKIP: sesión {history.get('status')}.")
        return

    spot, strikes = fetch_current_chain(today)
    snapshot = make_snapshot(now_iso, spot, strikes, history["legs"], history["entryCredit"])
    if not any(item.get("capturedAt") == now_iso for item in history["snapshots"]):
        history["snapshots"].append(snapshot)
    history["lastUpdated"] = now_iso
    save_json(history_file, history)
    update_index(today, now_iso)
    print(f"  OK: {len(history['snapshots'])} capturas; P&L estimado {snapshot['pnl']:.2f} USD")


def update_index(today, now_iso):
    index = load_json_safe(HISTORY_INDEX) or {"dates": []}
    if today not in index["dates"]:
        index["dates"].append(today)
        index["dates"].sort()
    index["lastUpdated"] = now_iso
    save_json(HISTORY_INDEX, index)


def main():
    now_madrid = datetime.now(ZoneInfo("Europe/Madrid"))
    today = now_madrid.date().isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()
    print(f"[{now_iso}] Premium monitor + intraday 0DTE chain (Madrid {now_madrid:%H:%M})")
    if not is_trading_day(now_madrid.date()):
        print("  SKIP: no es dia de mercado.")
        return
    if not (16 <= now_madrid.hour < 23):
        print("  SKIP: fuera de la ventana 16:00-22:59 Madrid.")
        return
    if recent_capture_exists(today, now_iso):
        print(f"  SKIP: ya existe una captura en los ultimos {MIN_CAPTURE_INTERVAL_MINUTES} minutos.")
        return

    spot, strikes = fetch_current_chain(today)
    chain_file = save_intraday_chain(today, now_iso, spot, strikes)
    print(f"  OK chain 0DTE: {chain_file} ({len(strikes)} strikes)")
    ensure_shadow_prediction(today, now_madrid)

    levels = load_trading_levels(today)
    if not levels:
        print(f"  WARN benchmark: no hay niveles operativos cargados para {today}.")
    else:
        try:
            history, history_file = ensure_benchmark_history(today, levels)
            update_index_file(HISTORY_INDEX, today, now_iso)
            snapshot = append_strategy_snapshot(history, history_file, now_iso, spot, strikes, HISTORY_INDEX)
            if snapshot:
                print(f"  OK benchmark: {len(history['snapshots'])} capturas; P&L {snapshot['pnl']:.2f} USD")
            else:
                print(f"  OK benchmark: sesion {history.get('status')}.")
        except RuntimeError as exc:
            if "cadena de entrada" in str(exc):
                print(f"  WARN benchmark: {exc}.")
            else:
                raise

    shadow_history, shadow_file = ensure_shadow_history(today)
    if shadow_history:
        update_index_file(SHADOW_HISTORY_INDEX, today, now_iso)
        shadow_snapshot = append_strategy_snapshot(shadow_history, shadow_file, now_iso, spot, strikes, SHADOW_HISTORY_INDEX)
        if shadow_snapshot:
            print(f"  OK ML Shadow: {len(shadow_history['snapshots'])} capturas; P&L {shadow_snapshot['pnl']:.2f} USD")
        else:
            print(f"  OK ML Shadow: sesion {shadow_history.get('status')}.")
    else:
        print("  WARN ML Shadow: prediccion no disponible todavia.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERR Premium monitor: {exc}", file=sys.stderr)
        raise
