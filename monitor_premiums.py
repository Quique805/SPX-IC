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
SPOTGAMMA_FILE = os.path.join(DATA_DIR, "spotgamma-levels.json")
CONTRACT_MULTIPLIER = 100
SPREAD_WIDTH = 20
MIN_WING_CREDIT_USD = 5
MIN_WING_CREDIT_POINTS = MIN_WING_CREDIT_USD / CONTRACT_MULTIPLIER
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
    }


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
        levels = load_spotgamma_levels(today)
        if not levels:
            raise RuntimeError(f"no hay niveles SpotGamma cargados para {today}")
        reason = no_trade_reason(today, levels)
        if reason:
            history = {
                "date": today,
                "status": "no_trade",
                "reason": reason,
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
            "sourceCloseDate": today,
            "levelsSource": levels.get("source"),
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


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERR Premium monitor: {exc}", file=sys.stderr)
        raise
