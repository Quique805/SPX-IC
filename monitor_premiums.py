#!/usr/bin/env python3
"""Capture intraday quotes for the four automatically selected SPX IC legs."""
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
OHLC_FILE = os.path.join(DATA_DIR, "daily-ohlc.json")
CLOSE_DIR = os.path.join(DATA_DIR, "chains-close")
ENTRY_DIR = os.path.join(DATA_DIR, "chains")
CONTRACT_MULTIPLIER = 100
SPREAD_WIDTH = 20
MIN_T = 1 / 252
RISK_FREE_RATE = 0.04


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
    grid = []
    for test_spot in [row["strike"] for row in rows if abs(row["strike"] / spot - 1) <= 0.08]:
        net = 0
        for raw in expiration.get("strikes", []):
            strike = float(raw.get("strike") or 0)
            call_iv = float(raw.get("call_iv") or raw.get("put_iv") or 0.18)
            put_iv = float(raw.get("put_iv") or raw.get("call_iv") or 0.18)
            net += exposure(test_spot, strike, call_iv, t, float(raw.get("call_oi") or 0), 1)
            net += exposure(test_spot, strike, put_iv, t, float(raw.get("put_oi") or 0), -1)
        grid.append((test_spot, net))
    trigger = None
    for previous, current in zip(grid, grid[1:]):
        if (previous[1] <= 0 <= current[1]) or (previous[1] >= 0 >= current[1]):
            denom = abs(previous[1]) + abs(current[1])
            weight = abs(previous[1]) / denom if denom else 0.5
            trigger = previous[0] + (current[0] - previous[0]) * weight
            break
    if trigger is None and grid:
        trigger = min(grid, key=lambda item: abs(item[1]))[0]
    return {"call_wall": call_wall, "put_wall": put_wall, "vol_trigger": trigger}


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


def no_trade_reason(today, levels):
    trigger = levels.get("vol_trigger")
    wall_range = levels["call_wall"] - levels["put_wall"]
    if trigger is not None and wall_range > 0:
        trigger_pct = (trigger - levels["put_wall"]) / wall_range * 100
        if trigger_pct < 15 or trigger_pct > 70:
            return f"Vol Trigger fuera de rango ({trigger_pct:.1f}%)"
    ohlc = (load_json_safe(OHLC_FILE) or {}).get("byDate", {})
    dates = sorted(date for date in ohlc if date < today and ohlc[date].get("spx_close") is not None)
    today_open = ohlc.get(today, {}).get("spx_open")
    if dates and today_open is not None:
        previous_close = ohlc[dates[-1]]["spx_close"]
        gap = (float(today_open) - float(previous_close)) / float(previous_close) * 100
        if abs(gap) >= 0.50:
            return f"Gap de apertura fuera de rango ({gap:+.2f}%)"
    return None


def select_legs(today, levels):
    entry = load_json_safe(os.path.join(ENTRY_DIR, f"{today}.json"))
    if not entry:
        raise RuntimeError("la cadena de entrada todavía no existe")
    _, expiration = nearest_expiration(entry)
    strikes = expiration.get("strikes", []) if expiration else []
    targets = {
        "sell_call": levels["call_wall"],
        "buy_call": levels["call_wall"] + SPREAD_WIDTH,
        "sell_put": levels["put_wall"],
        "buy_put": levels["put_wall"] - SPREAD_WIDTH,
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
    entry_credit = sum(
        leg["entryBid"] if leg["side"] == "sell" else -leg["entryAsk"]
        for leg in legs.values()
    )
    return entry, legs, entry_credit


def fetch_current_chain(today):
    raw = http_get_json("https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json")
    data = raw.get("data", {})
    spot = data.get("current_price") or data.get("last") or data.get("price")
    parsed = [parse_option(option) for option in data.get("options", [])]
    parsed = [option for option in parsed if option and option["expiration"] == today]
    return float(spot or 0), merge_strikes(parsed, today, float(spot or 0), max_pct=8.0)


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


def main():
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
        source_date, close_chain = previous_close_chain(today)
        levels = compute_levels(close_chain or {})
        if not levels:
            raise RuntimeError("no se pudieron calcular niveles desde la cadena de cierre anterior")
        reason = no_trade_reason(today, levels)
        if reason:
            history = {"date": today, "status": "no_trade", "reason": reason, "snapshots": []}
            save_json(history_file, history)
            print(f"  SKIP: {reason}")
            update_index(today, now_iso)
            return
        try:
            entry, legs, entry_credit = select_legs(today, levels)
        except RuntimeError as exc:
            if "cadena de entrada" in str(exc):
                print(f"  SKIP: {exc}. Se reintentará en la siguiente captura.")
                return
            raise
        history = {
            "date": today,
            "status": "active",
            "sourceCloseDate": source_date,
            "selectedAt": entry.get("capturedAt"),
            "entrySpot": entry.get("spot"),
            "entryCredit": entry_credit,
            "spreadWidth": SPREAD_WIDTH,
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


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERR Premium monitor: {exc}", file=sys.stderr)
        raise
