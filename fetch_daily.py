#!/usr/bin/env python3
"""
Daily SPX data fetch for the Iron Condor dashboard.
Fetches SPX & VIX OHLC from Yahoo + SPX option chain from CBOE.
Writes to data/daily-ohlc.json (accumulating) and data/chains/YYYY-MM-DD.json (one per day).
"""
import json, os, re, sys, urllib.request
from datetime import datetime, date, timezone

DATA_DIR = "data"
CHAINS_DIR = os.path.join(DATA_DIR, "chains")
OHLC_FILE = os.path.join(DATA_DIR, "daily-ohlc.json")
CHAINS_INDEX = os.path.join(DATA_DIR, "chains-index.json")
UA = "Mozilla/5.0 (compatible; SPX-IC-bot/1.0)"

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
        out.append({"date": d, "open": q["open"][i], "high": q["high"][i],
                    "low": q["low"][i], "close": q["close"][i]})
    return out

OPTION_RE = re.compile(r"^(SPXW?)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")
def parse_option(opt):
    m = OPTION_RE.match(opt.get("option", ""))
    if not m: return None
    _, yy, mm, dd, t, strike_str = m.groups()
    return {"expiration": f"20{yy}-{mm}-{dd}", "type": "call" if t == "C" else "put",
            "strike": int(strike_str) / 1000,
            "bid": opt.get("bid"), "ask": opt.get("ask"),
            "last": opt.get("last_trade_price"), "iv": opt.get("iv"),
            "volume": opt.get("volume"), "oi": opt.get("open_interest"),
            "delta": opt.get("delta")}

def merge_strikes(options, expiration, spot, max_pct=5.0):
    by_strike, lo, hi = {}, spot * (1 - max_pct/100), spot * (1 + max_pct/100)
    for opt in options:
        if opt is None or opt["expiration"] != expiration: continue
        s = opt["strike"]
        if s < lo or s > hi: continue
        if s not in by_strike: by_strike[s] = {"strike": s}
        prefix = "call" if opt["type"] == "call" else "put"
        for f in ("bid", "ask", "last", "iv", "volume", "oi", "delta"):
            by_strike[s][f"{prefix}_{f}"] = opt[f]
    return sorted(by_strike.values(), key=lambda x: x["strike"])

def load_json_safe(path):
    if not os.path.exists(path): return None
    try:
        with open(path) as f: return json.load(f)
    except Exception: return None

def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2, default=str)

def main():
    today = date.today().isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()
    print(f"[{now_iso}] SPX-IC daily fetch starting...")
    os.makedirs(CHAINS_DIR, exist_ok=True)

    # Yahoo OHLC
    spx_rows, vix_rows = [], []
    try:
        spx_rows = fetch_yahoo_ohlc("^GSPC")
        print(f"  ✓ Yahoo SPX: {len(spx_rows)} días")
    except Exception as e:
        print(f"  ✗ Yahoo SPX: {e}", file=sys.stderr)
    try:
        vix_rows = fetch_yahoo_ohlc("^VIX")
        print(f"  ✓ Yahoo VIX: {len(vix_rows)} días")
    except Exception as e:
        print(f"  ✗ Yahoo VIX: {e}", file=sys.stderr)

    if spx_rows or vix_rows:
        existing = load_json_safe(OHLC_FILE) or {"byDate": {}}
        by_date = existing.get("byDate", {})
        for row in spx_rows:
            d = row["date"]
            if d not in by_date: by_date[d] = {}
            by_date[d].update({"spx_open": row["open"], "spx_high": row["high"],
                               "spx_low": row["low"], "spx_close": row["close"]})
        for row in vix_rows:
            d = row["date"]
            if d not in by_date: by_date[d] = {}
            by_date[d].update({"vix_open": row["open"], "vix_close": row["close"]})
        save_json(OHLC_FILE, {"lastUpdated": now_iso, "byDate": by_date})
        print(f"  ✓ OHLC: {len(by_date)} fechas guardadas")

    # CBOE chain
    try:
        raw = http_get_json("https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json")
        d = raw.get("data", {})
        spot = d.get("current_price") or d.get("last") or d.get("price")
        options = [parse_option(o) for o in d.get("options", [])]
        options = [o for o in options if o is not None]
        print(f"  ✓ CBOE: spot={spot}, {len(options)} opciones")

        today_d = date.today()
        relevant = []
        for e in sorted(set(o["expiration"] for o in options)):
            ed = datetime.strptime(e, "%Y-%m-%d").date()
            dte = (ed - today_d).days
            if 0 <= dte <= 7: relevant.append((e, dte))

        chain_out = {"date": today, "capturedAt": now_iso, "spot": spot, "expirations": {}}
        for e, dte in relevant:
            chain_out["expirations"][e] = {"dte": dte,
                "strikes": merge_strikes(options, e, spot, max_pct=5.0)}
        chain_file = os.path.join(CHAINS_DIR, f"{today}.json")
        save_json(chain_file, chain_out)
        print(f"  ✓ Chain: {chain_file} ({len(relevant)} expiraciones)")

        index = load_json_safe(CHAINS_INDEX) or {"dates": []}
        if today not in index["dates"]:
            index["dates"].append(today); index["dates"].sort()
        index["lastUpdated"] = now_iso
        save_json(CHAINS_INDEX, index)
        print(f"  ✓ Índice: {len(index['dates'])} fechas con chain")
    except Exception as e:
        print(f"  ✗ CBOE: {e}", file=sys.stderr)

    print(f"[{datetime.now(timezone.utc).isoformat()}] Done.")

if __name__ == "__main__":
    main()
