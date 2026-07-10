#!/usr/bin/env python3
"""Generate experimental GEX model levels from saved close option chains."""
import json
import math
import os
from datetime import datetime, timezone

from fetch_daily import (
    CLOSE_CHAINS_DIR,
    CLOSE_CHAINS_INDEX,
    DATA_DIR,
    is_trading_day,
    load_json_safe,
    save_json,
)

OUT_DIR = os.path.join(DATA_DIR, "gex-models")
OUT_INDEX = os.path.join(DATA_DIR, "gex-models-index.json")
OHLC_FILE = os.path.join(DATA_DIR, "daily-ohlc.json")
SPOTGAMMA_FILE = os.path.join(DATA_DIR, "spotgamma-levels.json")

RISK_FREE_RATE = 0.045
CONTRACT_MULT = 100
MIN_T = 1 / 252
STRIKE_ROUNDING = 25
MODEL_VERSION = "gex-lab-v1"

MODEL_INFO = {
    "pure_gex": {
        "name": "Pure GEX",
        "description": "Lectura directa de concentracion gamma por strike, ponderada por vencimiento.",
    },
    "liquidity_weighted": {
        "name": "Liquidity Weighted GEX",
        "description": "Gamma ajustada por liquidez, volumen, open interest y proximidad al spot.",
    },
    "spotgamma_fit": {
        "name": "SpotGamma Fit",
        "description": "Modelo calibrado con el sesgo historico frente a los niveles SpotGamma manuales.",
    },
}


def normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_gamma(spot, strike, sigma, years_to_expiry):
    if not (spot > 0 and strike > 0):
        return 0.0
    vol = min(max(float(sigma) if sigma and sigma > 0 else 0.18, 0.03), 5.0)
    t = max(float(years_to_expiry) if years_to_expiry else 0.0, MIN_T)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    return normal_pdf(d1) / (spot * vol * sqrt_t)


def exposure(spot, strike, sigma, years_to_expiry, oi, sign):
    oi = max(float(oi or 0), 0)
    if oi <= 0:
        return 0.0
    return sign * bs_gamma(spot, strike, sigma, years_to_expiry) * oi * CONTRACT_MULT * spot * spot * 0.01


def trading_next(date_str):
    from datetime import date, timedelta

    y, m, d = [int(x) for x in date_str.split("-")]
    cur = date(y, m, d)
    while True:
        cur += timedelta(days=1)
        if is_trading_day(cur):
            return cur.isoformat()


def round_strike(value, step=STRIKE_ROUNDING):
    if value is None or not math.isfinite(float(value)):
        return None
    return int(round(float(value) / step) * step)


def safe_num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def dte_weight(dte):
    return 1 / math.sqrt(max(float(dte or 0), 1))


def selected_expirations(chain, mode):
    expirations = []
    for exp, data in (chain.get("expirations") or {}).items():
        dte = safe_num(data.get("dte"), 0)
        strikes = data.get("strikes") or []
        if dte <= 0 or not strikes:
            continue
        expirations.append({"exp": exp, "dte": dte, "strikes": strikes})
    expirations.sort(key=lambda item: item["dte"])
    if mode == "next":
        return expirations[:1]
    return expirations


def aggregate_rows(chain, mode, variant):
    spot = safe_num(chain.get("spot"))
    rows_by_strike = {}
    expirations = selected_expirations(chain, mode)
    for exp in expirations:
        weight = 1.0 if mode == "next" else dte_weight(exp["dte"])
        years = max(exp["dte"] / 252, MIN_T)
        for raw in exp["strikes"]:
            strike = safe_num(raw.get("strike"))
            if strike <= 0:
                continue
            call_iv = safe_num(raw.get("call_iv"), safe_num(raw.get("put_iv"), 0.18))
            put_iv = safe_num(raw.get("put_iv"), safe_num(raw.get("call_iv"), 0.18))
            call_oi = safe_num(raw.get("call_oi"))
            put_oi = safe_num(raw.get("put_oi"))
            call_volume = safe_num(raw.get("call_volume"))
            put_volume = safe_num(raw.get("put_volume"))
            call_gex = exposure(spot, strike, call_iv, years, call_oi, 1) * weight
            put_gex = exposure(spot, strike, put_iv, years, put_oi, -1) * weight
            row = rows_by_strike.setdefault(strike, {
                "strike": strike,
                "callGex": 0.0,
                "putGex": 0.0,
                "callOi": 0.0,
                "putOi": 0.0,
                "callVolume": 0.0,
                "putVolume": 0.0,
                "expirations": 0,
            })
            row["callGex"] += call_gex
            row["putGex"] += put_gex
            row["callOi"] += call_oi * weight
            row["putOi"] += put_oi * weight
            row["callVolume"] += call_volume * weight
            row["putVolume"] += put_volume * weight
            row["expirations"] += 1

    rows = []
    for row in rows_by_strike.values():
        strike = row["strike"]
        distance = abs(strike - spot) / spot if spot > 0 else 0
        proximity = max(0.05, 1 - min(distance / 0.08, 0.95))
        call_liquidity = math.log1p(row["callOi"]) + 0.55 * math.log1p(row["callVolume"])
        put_liquidity = math.log1p(row["putOi"]) + 0.55 * math.log1p(row["putVolume"])
        if variant == "liquidity_weighted":
            call_score = math.log1p(abs(row["callGex"])) * (1 + 0.10 * call_liquidity) * proximity
            put_score = math.log1p(abs(row["putGex"])) * (1 + 0.10 * put_liquidity) * proximity
        else:
            call_score = abs(row["callGex"])
            put_score = abs(row["putGex"])
        row["netGex"] = row["callGex"] + row["putGex"]
        row["absGex"] = abs(row["callGex"]) + abs(row["putGex"])
        row["callScore"] = call_score
        row["putScore"] = put_score
        rows.append(row)
    rows.sort(key=lambda item: item["strike"])
    return rows, expirations


def option_book(chain, mode):
    spot = safe_num(chain.get("spot"))
    book = []
    for exp in selected_expirations(chain, mode):
        weight = 1.0 if mode == "next" else dte_weight(exp["dte"])
        years = max(exp["dte"] / 252, MIN_T)
        for raw in exp["strikes"]:
            strike = safe_num(raw.get("strike"))
            if strike <= 0:
                continue
            book.append({
                "strike": strike,
                "years": years,
                "weight": weight,
                "callIv": safe_num(raw.get("call_iv"), safe_num(raw.get("put_iv"), 0.18)),
                "putIv": safe_num(raw.get("put_iv"), safe_num(raw.get("call_iv"), 0.18)),
                "callOi": safe_num(raw.get("call_oi")),
                "putOi": safe_num(raw.get("put_oi")),
            })
    return spot, book


def total_net_gex_at(test_spot, book):
    total = 0.0
    for opt in book:
        total += exposure(test_spot, opt["strike"], opt["callIv"], opt["years"], opt["callOi"], 1) * opt["weight"]
        total += exposure(test_spot, opt["strike"], opt["putIv"], opt["years"], opt["putOi"], -1) * opt["weight"]
    return total


def curve_levels(chain, mode):
    spot, book = option_book(chain, mode)
    if not book or spot <= 0:
        return None, None, []
    lo = round_strike(spot * 0.93)
    hi = round_strike(spot * 1.07)
    grid = list(range(int(lo), int(hi) + STRIKE_ROUNDING, STRIKE_ROUNDING))
    curve = [{"spot": s, "netGex": total_net_gex_at(s, book)} for s in grid]
    if not curve:
        return None, None, []

    crossings = []
    for prev, cur in zip(curve, curve[1:]):
        if prev["netGex"] == 0:
            crossings.append(prev["spot"])
        elif prev["netGex"] * cur["netGex"] < 0:
            ratio = abs(prev["netGex"]) / (abs(prev["netGex"]) + abs(cur["netGex"]))
            crossings.append(prev["spot"] + (cur["spot"] - prev["spot"]) * ratio)
    if crossings:
        gamma_flip = min(crossings, key=lambda s: abs(s - spot))
    else:
        gamma_flip = min(curve, key=lambda p: abs(p["netGex"]))["spot"]

    slopes = []
    for prev, cur in zip(curve, curve[1:]):
        mid = (prev["spot"] + cur["spot"]) / 2
        if abs(mid - spot) <= spot * 0.06:
            slopes.append({"spot": mid, "slope": abs(cur["netGex"] - prev["netGex"])})
    vol_trigger = max(slopes, key=lambda p: p["slope"])["spot"] if slopes else gamma_flip
    return round_strike(gamma_flip), round_strike(vol_trigger), curve


def choose_walls(rows, spot, variant):
    if not rows:
        return None, None
    calls = [r for r in rows if r["strike"] >= spot] or rows
    puts = [r for r in rows if r["strike"] <= spot] or rows
    if variant == "liquidity_weighted":
        call = max(calls, key=lambda r: r["callScore"])
        put = max(puts, key=lambda r: r["putScore"])
    else:
        call = max(calls, key=lambda r: abs(r["callGex"]))
        put = max(puts, key=lambda r: abs(r["putGex"]))
    return round_strike(call["strike"]), round_strike(put["strike"])


def compute_variant(chain, mode, variant):
    rows, expirations = aggregate_rows(chain, mode, variant)
    spot = safe_num(chain.get("spot"))
    call_wall, put_wall = choose_walls(rows, spot, variant)
    gamma_flip, vol_trigger, curve = curve_levels(chain, mode)
    top_rows = sorted(rows, key=lambda r: r["absGex"], reverse=True)[:12]
    return {
        "callWall": call_wall,
        "putWall": put_wall,
        "gammaFlip": gamma_flip,
        "volTrigger": vol_trigger,
        "netAtSpot": total_net_gex_at(spot, option_book(chain, mode)[1]) if spot > 0 else None,
        "expirationsUsed": [{"expiration": e["exp"], "dte": e["dte"]} for e in expirations],
        "histogram": [{
            "strike": round(r["strike"], 2),
            "callGex": r["callGex"],
            "putGex": r["putGex"],
            "netGex": r["netGex"],
            "absGex": r["absGex"],
            "callScore": r["callScore"],
            "putScore": r["putScore"],
        } for r in rows],
        "topRows": top_rows,
        "curve": curve,
    }


def reference_for_target(spotgamma, target_date):
    row = ((spotgamma or {}).get("byDate") or {}).get(target_date)
    if not row:
        return None
    return {
        "callWall": safe_num(row.get("callWall"), None),
        "putWall": safe_num(row.get("putWall"), None),
        "gammaFlip": safe_num(row.get("gammaFlip"), None) if row.get("gammaFlip") is not None else None,
        "volTrigger": safe_num(row.get("volTrigger"), None) if row.get("volTrigger") is not None else None,
        "source": row.get("source") or "SpotGamma manual",
        "updatedAt": row.get("updatedAt"),
    }


def level_errors(levels, reference):
    if not levels or not reference:
        return None
    weights = {"callWall": 0.35, "putWall": 0.35, "gammaFlip": 0.15, "volTrigger": 0.15}
    errors = {}
    weighted = 0.0
    used = 0.0
    for key, weight in weights.items():
        ref = reference.get(key)
        pred = levels.get(key)
        if ref is None or pred is None:
            errors[key] = None
            continue
        err = abs(float(pred) - float(ref))
        errors[key] = err
        weighted += err * weight
        used += weight
    errors["weightedError"] = weighted / used if used else None
    return errors


def operation_diagnostic(levels, target_date, ohlc):
    row = ((ohlc or {}).get("byDate") or {}).get(target_date) or {}
    call_wall = safe_num(levels.get("callWall"), None)
    put_wall = safe_num(levels.get("putWall"), None)
    open_ = safe_num(row.get("spx_open"), None)
    high = safe_num(row.get("spx_high"), None)
    low = safe_num(row.get("spx_low"), None)
    if call_wall is None or put_wall is None or open_ is None or call_wall <= put_wall:
        return {"status": "pending", "reason": "Datos insuficientes"}
    open_pct = (open_ - put_wall) / (call_wall - put_wall) * 100
    if open_pct < 10 or open_pct > 90:
        return {"status": "no_trade", "reason": "Open fuera de rango", "openPct": open_pct}
    sell_call = call_wall
    sell_put = put_wall
    adjustment = "Sin ajuste"
    if 10 <= open_pct <= 20:
        sell_put = put_wall - 35
        adjustment = "Put Wall -35"
    elif 20 < open_pct <= 30:
        sell_put = put_wall - 20
        adjustment = "Put Wall -20"
    elif 80 < open_pct <= 90:
        sell_call = call_wall + 15
        adjustment = "Call Wall +15"
    if high is None or low is None:
        return {
            "status": "pending",
            "reason": "OHLC incompleto",
            "openPct": open_pct,
            "sellCall": sell_call,
            "sellPut": sell_put,
            "adjustment": adjustment,
        }
    upper_touch = high >= sell_call
    lower_touch = low <= sell_put
    return {
        "status": "loss" if upper_touch or lower_touch else "win",
        "openPct": open_pct,
        "sellCall": sell_call,
        "sellPut": sell_put,
        "adjustment": adjustment,
        "upperTouch": upper_touch,
        "lowerTouch": lower_touch,
        "high": high,
        "low": low,
    }


def median(values):
    clean = sorted(v for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return 0
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2


def apply_spotgamma_fit(records):
    prior_errors = {mode: {key: [] for key in ("callWall", "putWall", "gammaFlip", "volTrigger")} for mode in ("next", "multi")}
    for record in records:
        model = {"name": MODEL_INFO["spotgamma_fit"]["name"], "description": MODEL_INFO["spotgamma_fit"]["description"]}
        for mode in ("next", "multi"):
            base = record["models"]["liquidity_weighted"][mode]
            fitted = json.loads(json.dumps(base))
            for key in ("callWall", "putWall", "gammaFlip", "volTrigger"):
                if fitted.get(key) is not None:
                    fitted[key] = round_strike(fitted[key] + median(prior_errors[mode][key]))
            model[mode] = fitted
        record["models"]["spotgamma_fit"] = model

        ref = record.get("reference", {}).get("spotgamma")
        if ref:
            for mode in ("next", "multi"):
                base = record["models"]["liquidity_weighted"][mode]
                for key in ("callWall", "putWall", "gammaFlip", "volTrigger"):
                    if base.get(key) is not None and ref.get(key) is not None:
                        prior_errors[mode][key].append(ref[key] - base[key])


def build_record(chain_date, chain, spotgamma, ohlc):
    target_date = trading_next(chain_date)
    record = {
        "version": MODEL_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "date": chain_date,
        "targetSession": target_date,
        "sourceChain": f"data/chains-close/{chain_date}.json",
        "spot": chain.get("spot"),
        "capturedAt": chain.get("capturedAt"),
        "models": {},
        "reference": {"spotgamma": reference_for_target(spotgamma, target_date)},
        "diagnostics": {},
    }
    for variant in ("pure_gex", "liquidity_weighted"):
        record["models"][variant] = {
            "name": MODEL_INFO[variant]["name"],
            "description": MODEL_INFO[variant]["description"],
            "next": compute_variant(chain, "next", variant),
            "multi": compute_variant(chain, "multi", variant),
        }
    return record


def attach_diagnostics(record, ohlc):
    reference = (record.get("reference") or {}).get("spotgamma")
    diagnostics = {}
    for model_id, model in record["models"].items():
        diagnostics[model_id] = {}
        for mode in ("next", "multi"):
            levels = model[mode]
            diagnostics[model_id][mode] = {
                "spotgammaError": level_errors(levels, reference),
                "operation": operation_diagnostic(levels, record["targetSession"], ohlc),
            }
    record["diagnostics"] = diagnostics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    close_index = load_json_safe(CLOSE_CHAINS_INDEX) or {"dates": []}
    spotgamma = load_json_safe(SPOTGAMMA_FILE) or {"byDate": {}}
    ohlc = load_json_safe(OHLC_FILE) or {"byDate": {}}
    records = []
    for chain_date in sorted(close_index.get("dates", [])):
        path = os.path.join(CLOSE_CHAINS_DIR, f"{chain_date}.json")
        chain = load_json_safe(path)
        if not chain:
            continue
        try:
            record = build_record(chain_date, chain, spotgamma, ohlc)
            if not record["models"]["pure_gex"]["next"]["expirationsUsed"]:
                continue
            records.append(record)
        except Exception as exc:
            print(f"WARN {chain_date}: {exc}")
    apply_spotgamma_fit(records)
    for record in records:
        attach_diagnostics(record, ohlc)
        save_json(os.path.join(OUT_DIR, f"{record['date']}.json"), record)
    valid_files = {f"{record['date']}.json" for record in records}
    for name in os.listdir(OUT_DIR):
        if name.endswith(".json") and name not in valid_files:
            os.remove(os.path.join(OUT_DIR, name))
    index = {
        "version": MODEL_VERSION,
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "dates": [record["date"] for record in records],
        "models": MODEL_INFO,
    }
    save_json(OUT_INDEX, index)
    print(f"OK GEX models: {len(records)} dates")


if __name__ == "__main__":
    main()
