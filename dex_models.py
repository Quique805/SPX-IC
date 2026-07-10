#!/usr/bin/env python3
"""Generate experimental DEX levels from saved close option chains."""
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

OUT_DIR = os.path.join(DATA_DIR, "dex-models")
OUT_INDEX = os.path.join(DATA_DIR, "dex-models-index.json")

RISK_FREE_RATE = 0.045
CONTRACT_MULT = 100
MIN_T = 1 / 252
STRIKE_ROUNDING = 25
MODEL_VERSION = "dex-lab-v1"

MODEL_INFO = {
    "weighted_dex": {
        "name": "Weighted DEX",
        "description": "Delta exposure ponderada por vencimiento: el vencimiento cercano manda, los siguientes aportan contexto decreciente.",
    }
}


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def safe_num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def round_strike(value, step=STRIKE_ROUNDING):
    if value is None or not math.isfinite(float(value)):
        return None
    return int(round(float(value) / step) * step)


def trading_next(date_str):
    from datetime import date, timedelta

    y, m, d = [int(x) for x in date_str.split("-")]
    cur = date(y, m, d)
    while True:
        cur += timedelta(days=1)
        if is_trading_day(cur):
            return cur.isoformat()


def dte_weight(dte):
    return 1 / math.sqrt(max(float(dte or 0), 1))


def selected_expirations(chain):
    expirations = []
    for exp, data in (chain.get("expirations") or {}).items():
        dte = safe_num(data.get("dte"), 0)
        strikes = data.get("strikes") or []
        if dte <= 0 or not strikes:
            continue
        expirations.append({"exp": exp, "dte": dte, "strikes": strikes})
    expirations.sort(key=lambda item: item["dte"])
    return expirations


def clean_iv(primary, fallback=0.18):
    iv = safe_num(primary, 0)
    if iv <= 0.01 or iv > 3.0:
        iv = safe_num(fallback, 0.18)
    if iv <= 0.01 or iv > 3.0:
        iv = 0.18
    return min(max(iv, 0.03), 3.0)


def bs_delta(spot, strike, sigma, years_to_expiry, side):
    if not (spot > 0 and strike > 0):
        return 0.0
    vol = clean_iv(sigma)
    t = max(float(years_to_expiry) if years_to_expiry else 0.0, MIN_T)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    call_delta = normal_cdf(d1)
    return call_delta if side == "call" else call_delta - 1


def raw_or_bs_delta(raw, spot, strike, sigma, years, side):
    delta = safe_num(raw, None)
    if delta is not None and -1.01 <= delta <= 1.01:
        return delta
    return bs_delta(spot, strike, sigma, years, side)


def dex_exposure(delta, oi, spot):
    oi = max(safe_num(oi), 0)
    if oi <= 0:
        return 0.0
    return delta * oi * CONTRACT_MULT * spot


def aggregate_rows(chain):
    spot = safe_num(chain.get("spot"))
    rows_by_strike = {}
    expirations = selected_expirations(chain)
    for exp in expirations:
        weight = dte_weight(exp["dte"])
        years = max(exp["dte"] / 252, MIN_T)
        for raw in exp["strikes"]:
            strike = safe_num(raw.get("strike"))
            if strike <= 0:
                continue
            call_iv = clean_iv(raw.get("call_iv"), raw.get("put_iv"))
            put_iv = clean_iv(raw.get("put_iv"), raw.get("call_iv"))
            call_delta = raw_or_bs_delta(raw.get("call_delta"), spot, strike, call_iv, years, "call")
            put_delta = raw_or_bs_delta(raw.get("put_delta"), spot, strike, put_iv, years, "put")
            call_oi = safe_num(raw.get("call_oi"))
            put_oi = safe_num(raw.get("put_oi"))
            call_dex = dex_exposure(call_delta, call_oi, spot) * weight
            put_dex = dex_exposure(put_delta, put_oi, spot) * weight
            row = rows_by_strike.setdefault(strike, {
                "strike": strike,
                "callDex": 0.0,
                "putDex": 0.0,
                "callOi": 0.0,
                "putOi": 0.0,
                "callVolume": 0.0,
                "putVolume": 0.0,
                "expirations": 0,
            })
            row["callDex"] += call_dex
            row["putDex"] += put_dex
            row["callOi"] += call_oi * weight
            row["putOi"] += put_oi * weight
            row["callVolume"] += safe_num(raw.get("call_volume")) * weight
            row["putVolume"] += safe_num(raw.get("put_volume")) * weight
            row["expirations"] += 1

    rows = []
    for row in rows_by_strike.values():
        row["netDex"] = row["callDex"] + row["putDex"]
        row["absDex"] = abs(row["callDex"]) + abs(row["putDex"])
        rows.append(row)
    rows.sort(key=lambda item: item["strike"])
    return rows, expirations


def option_book(chain):
    book = []
    for exp in selected_expirations(chain):
        weight = dte_weight(exp["dte"])
        years = max(exp["dte"] / 252, MIN_T)
        for raw in exp["strikes"]:
            strike = safe_num(raw.get("strike"))
            if strike <= 0:
                continue
            book.append({
                "strike": strike,
                "years": years,
                "weight": weight,
                "callIv": clean_iv(raw.get("call_iv"), raw.get("put_iv")),
                "putIv": clean_iv(raw.get("put_iv"), raw.get("call_iv")),
                "callOi": safe_num(raw.get("call_oi")),
                "putOi": safe_num(raw.get("put_oi")),
            })
    return book


def total_net_dex_at(test_spot, book):
    total = 0.0
    for opt in book:
        call_delta = bs_delta(test_spot, opt["strike"], opt["callIv"], opt["years"], "call")
        put_delta = bs_delta(test_spot, opt["strike"], opt["putIv"], opt["years"], "put")
        total += dex_exposure(call_delta, opt["callOi"], test_spot) * opt["weight"]
        total += dex_exposure(put_delta, opt["putOi"], test_spot) * opt["weight"]
    return total


def curve_levels(chain):
    spot = safe_num(chain.get("spot"))
    book = option_book(chain)
    if not book or spot <= 0:
        return None, None, None, []
    lo = round_strike(spot * 0.93)
    hi = round_strike(spot * 1.07)
    grid = list(range(int(lo), int(hi) + STRIKE_ROUNDING, STRIKE_ROUNDING))
    curve = [{"spot": s, "netDex": total_net_dex_at(s, book)} for s in grid]
    if not curve:
        return None, None, None, []

    crossings = []
    for prev, cur in zip(curve, curve[1:]):
        if prev["netDex"] == 0:
            crossings.append(prev["spot"])
        elif prev["netDex"] * cur["netDex"] < 0:
            ratio = abs(prev["netDex"]) / (abs(prev["netDex"]) + abs(cur["netDex"]))
            crossings.append(prev["spot"] + (cur["spot"] - prev["spot"]) * ratio)
    if crossings:
        dex_flip = min(crossings, key=lambda s: abs(s - spot))
    else:
        dex_flip = min(curve, key=lambda p: abs(p["netDex"]))["spot"]

    near = [p for p in curve if abs(p["spot"] - spot) <= spot * 0.06] or curve
    max_positive = max(near, key=lambda p: p["netDex"])
    max_negative = min(near, key=lambda p: p["netDex"])
    return round_strike(dex_flip), round_strike(max_positive["spot"]), round_strike(max_negative["spot"]), curve


def choose_delta_walls(rows, spot):
    if not rows:
        return None, None
    calls = [r for r in rows if r["strike"] >= spot] or rows
    puts = [r for r in rows if r["strike"] <= spot] or rows
    call = max(calls, key=lambda r: abs(r["callDex"]))
    put = max(puts, key=lambda r: abs(r["putDex"]))
    return round_strike(call["strike"]), round_strike(put["strike"])


def build_record(chain_date, chain):
    rows, expirations = aggregate_rows(chain)
    spot = safe_num(chain.get("spot"))
    call_wall, put_wall = choose_delta_walls(rows, spot)
    dex_flip, max_positive, max_negative, curve = curve_levels(chain)
    net_at_spot = total_net_dex_at(spot, option_book(chain)) if spot > 0 else None
    return {
        "version": MODEL_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "date": chain_date,
        "targetSession": trading_next(chain_date),
        "sourceChain": f"data/chains-close/{chain_date}.json",
        "spot": chain.get("spot"),
        "capturedAt": chain.get("capturedAt"),
        "model": MODEL_INFO["weighted_dex"],
        "levels": {
            "callDeltaWall": call_wall,
            "putDeltaWall": put_wall,
            "dexFlip": dex_flip,
            "maxPositiveDex": max_positive,
            "maxNegativeDex": max_negative,
            "netDexAtSpot": net_at_spot,
            "expirationsUsed": [{"expiration": e["exp"], "dte": e["dte"], "weight": dte_weight(e["dte"])} for e in expirations],
            "histogram": [{
                "strike": round(r["strike"], 2),
                "callDex": r["callDex"],
                "putDex": r["putDex"],
                "netDex": r["netDex"],
                "absDex": r["absDex"],
            } for r in rows],
            "topRows": sorted(rows, key=lambda r: r["absDex"], reverse=True)[:12],
            "curve": curve,
        },
        "diagnostics": {
            "regime": "positive" if net_at_spot and net_at_spot > 0 else "negative" if net_at_spot and net_at_spot < 0 else "neutral",
        },
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    close_index = load_json_safe(CLOSE_CHAINS_INDEX) or {"dates": []}
    records = []
    for chain_date in sorted(close_index.get("dates", [])):
        path = os.path.join(CLOSE_CHAINS_DIR, f"{chain_date}.json")
        chain = load_json_safe(path)
        if not chain:
            continue
        try:
            record = build_record(chain_date, chain)
            if not record["levels"]["expirationsUsed"]:
                continue
            records.append(record)
        except Exception as exc:
            print(f"WARN {chain_date}: {exc}")

    for record in records:
        save_json(os.path.join(OUT_DIR, f"{record['date']}.json"), record)
    valid_files = {f"{record['date']}.json" for record in records}
    for name in os.listdir(OUT_DIR):
        if name.endswith(".json") and name not in valid_files:
            os.remove(os.path.join(OUT_DIR, name))

    index = {
        "version": MODEL_VERSION,
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "dates": [record["date"] for record in records],
        "model": MODEL_INFO["weighted_dex"],
    }
    save_json(OUT_INDEX, index)
    print(f"OK DEX models: {len(records)} dates")


if __name__ == "__main__":
    main()
