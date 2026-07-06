#!/usr/bin/env python3
"""
Daily SPX data fetch for the Iron Condor dashboard.
Fetches SPX & VIX OHLC from Yahoo + SPX option chains from CBOE.

Outputs:
- data/daily-ohlc.json                 accumulated OHLC
- data/chains/YYYY-MM-DD.json          entry snapshot (~16:07 Madrid)
- data/chains-close/YYYY-MM-DD.json    close snapshot (~22:32 Madrid)
"""
import json, os, re, sys, urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

DATA_DIR = "data"
CHAINS_DIR = os.path.join(DATA_DIR, "chains")
CLOSE_CHAINS_DIR = os.path.join(DATA_DIR, "chains-close")
OHLC_FILE = os.path.join(DATA_DIR, "daily-ohlc.json")
CHAINS_INDEX = os.path.join(DATA_DIR, "chains-index.json")
CLOSE_CHAINS_INDEX = os.path.join(DATA_DIR, "chains-close-index.json")
UA = "Mozilla/5.0 (compatible; SPX-IC-bot/1.0)"

# NYSE full-close holidays. Update yearly.
NYSE_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}

def is_trading_day(d):
    """True if d (datetime.date) is a NYSE trading day (Mon-Fri, not a holiday)."""
    if d.weekday() >= 5:
        return False
    if d.isoformat() in NYSE_HOLIDAYS:
        return False
    return True

def http_get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def fetch_yahoo_ohlc(symbol, range_="10d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval=1d"
    data = http_get_json(url)
    res = data["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    out = []
    for i, t in enumerate(ts):
        d = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        out.append({
            "date": d,
            "open": q["open"][i],
            "high": q["high"][i],
            "low": q["low"][i],
            "close": q["close"][i],
        })
    return out

OPTION_RE = re.compile(r"^(SPXW?)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")

def parse_option(opt):
    m = OPTION_RE.match(opt.get("option", ""))
    if not m:
        return None
    _, yy, mm, dd, t, strike_str = m.groups()
    return {
        "expiration": f"20{yy}-{mm}-{dd}",
        "type": "call" if t == "C" else "put",
        "strike": int(strike_str) / 1000,
        "bid": opt.get("bid"),
        "ask": opt.get("ask"),
        "last": opt.get("last_trade_price"),
        "iv": opt.get("iv"),
        "volume": opt.get("volume"),
        "oi": opt.get("open_interest"),
        "delta": opt.get("delta"),
    }

def merge_strikes(options, expiration, spot, max_pct=5.0):
    by_strike = {}
    lo, hi = spot * (1 - max_pct / 100), spot * (1 + max_pct / 100)
    for opt in options:
        if opt is None or opt["expiration"] != expiration:
            continue
        s = opt["strike"]
        if s < lo or s > hi:
            continue
        if s not in by_strike:
            by_strike[s] = {"strike": s}
        prefix = "call" if opt["type"] == "call" else "put"
        for f in ("bid", "ask", "last", "iv", "volume", "oi", "delta"):
            by_strike[s][f"{prefix}_{f}"] = opt[f]
    return sorted(by_strike.values(), key=lambda x: x["strike"])

def load_json_safe(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def madrid_phase(now_madrid):
    """Return open/entry/close/off based on Madrid market workflow windows."""
    if now_madrid.hour < 16:
        return "open"
    if now_madrid.hour >= 22:
        return "close"
    return "entry"

def main():
    now_madrid = datetime.now(ZoneInfo("Europe/Madrid"))
    today_d = now_madrid.date()
    today = today_d.isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()
    phase = madrid_phase(now_madrid)
    print(f"[{now_iso}] SPX-IC fetch starting (today={today}, madrid={now_madrid:%H:%M}, phase={phase})")

    if not is_trading_day(today_d):
        reason = "weekend" if today_d.weekday() >= 5 else "NYSE holiday"
        print(f"  SKIP: today is {reason}. No data fetched.")
        return

    os.makedirs(CHAINS_DIR, exist_ok=True)
    os.makedirs(CLOSE_CHAINS_DIR, exist_ok=True)

    # Yahoo OHLC
    spx_rows, vix_rows = [], []
    try:
        spx_rows = fetch_yahoo_ohlc("^GSPC")
        print(f"  OK Yahoo SPX: {len(spx_rows)} days")
    except Exception as e:
        print(f"  ERR Yahoo SPX: {e}", file=sys.stderr)
    try:
        vix_rows = fetch_yahoo_ohlc("^VIX")
        print(f"  OK Yahoo VIX: {len(vix_rows)} days")
    except Exception as e:
        print(f"  ERR Yahoo VIX: {e}", file=sys.stderr)

    def r_int(v):
        return round(v) if v is not None else None
    def r_vix(v):
        return round(v, 1) if v is not None else None

    if spx_rows or vix_rows:
        existing = load_json_safe(OHLC_FILE) or {"byDate": {}}
        by_date = existing.get("byDate", {})
        for row in spx_rows:
            d = row["date"]
            if d not in by_date:
                by_date[d] = {}
            new_open = r_int(row["open"])
            if by_date[d].get("spx_open") is None and new_open is not None:
                by_date[d]["spx_open"] = new_open
            if row["high"] is not None:
                by_date[d]["spx_high"] = r_int(row["high"])
            if row["low"] is not None:
                by_date[d]["spx_low"] = r_int(row["low"])
            if row["close"] is not None:
                by_date[d]["spx_close"] = r_int(row["close"])
        for row in vix_rows:
            d = row["date"]
            if d not in by_date:
                by_date[d] = {}
            new_vix_open = r_vix(row["open"])
            if by_date[d].get("vix_open") is None and new_vix_open is not None:
                by_date[d]["vix_open"] = new_vix_open
            if row["close"] is not None:
                by_date[d]["vix_close"] = r_vix(row["close"])
        save_json(OHLC_FILE, {"lastUpdated": now_iso, "byDate": by_date})
        print(f"  OK OHLC: {len(by_date)} dates saved")

    # Cross-check: if Yahoo does not have today's daily bar yet, market is not open.
    latest_spx = spx_rows[-1]["date"] if spx_rows else None
    if latest_spx and latest_spx != today:
        print(f"  SKIP chain: latest SPX bar is {latest_spx}, not {today}.")
        print(f"[{datetime.now(timezone.utc).isoformat()}] Done.")
        return

    if phase == "open":
        print(f"  SKIP chain: Madrid {now_madrid:%H:%M} is Open phase. CBOE delayed data would be pre-market.")
        print(f"[{datetime.now(timezone.utc).isoformat()}] Done.")
        return

    chain_kind = "close" if phase == "close" else "entry"
    chain_dir = CLOSE_CHAINS_DIR if chain_kind == "close" else CHAINS_DIR
    chain_index_file = CLOSE_CHAINS_INDEX if chain_kind == "close" else CHAINS_INDEX
    chain_file = os.path.join(chain_dir, f"{today}.json")

    if os.path.exists(chain_file):
        print(f"  SKIP chain: {chain_kind} snapshot already exists ({chain_file}).")
        print(f"[{datetime.now(timezone.utc).isoformat()}] Done.")
        return

    # CBOE chain
    try:
        raw = http_get_json("https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json")
        d = raw.get("data", {})
        spot = d.get("current_price") or d.get("last") or d.get("price")
        options = [parse_option(o) for o in d.get("options", [])]
        options = [o for o in options if o is not None]
        print(f"  OK CBOE: spot={spot}, {len(options)} options")

        relevant = []
        for e in sorted(set(o["expiration"] for o in options)):
            ed = datetime.strptime(e, "%Y-%m-%d").date()
            dte = (ed - today_d).days
            if dte >= 0:
                relevant.append((e, dte))
        relevant = relevant[:5]

        chain_out = {"date": today, "kind": chain_kind, "capturedAt": now_iso, "spot": spot, "expirations": {}}
        for e, dte in relevant:
            chain_out["expirations"][e] = {
                "dte": dte,
                "strikes": merge_strikes(options, e, spot, max_pct=5.0),
            }
        save_json(chain_file, chain_out)
        print(f"  OK Chain {chain_kind}: {chain_file} ({len(relevant)} expirations)")

        index = load_json_safe(chain_index_file) or {"dates": []}
        if today not in index["dates"]:
            index["dates"].append(today)
            index["dates"].sort()
        index["lastUpdated"] = now_iso
        save_json(chain_index_file, index)
        print(f"  OK Index {chain_kind}: {len(index['dates'])} chain dates")
    except Exception as e:
        print(f"  ERR CBOE: {e}", file=sys.stderr)

    print(f"[{datetime.now(timezone.utc).isoformat()}] Done.")

if __name__ == "__main__":
    main()
