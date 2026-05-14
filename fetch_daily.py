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
    if d.weekday() >= 5: return False  # 5=Sat, 6=Sun
    if d.isoformat() in NYSE_HOLIDAYS: return False
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
    today_d = date.today()
    today = today_d.isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()
    print(f"[{now_iso}] SPX-IC daily fetch starting (today={today}, weekday={today_d.strftime('%A')})...")

    if not is_trading_day(today_d):
        reason = "fin de semana" if today_d.weekday() >= 5 else "festivo NYSE"
        print(f"  ⏭  Saltando: hoy es {reason}. No se descarga nada.")
        return

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

    def r_int(v): return round(v) if v is not None else None  # SPX → entero
    def r_vix(v): return round(v, 1) if v is not None else None  # VIX → 1 decimal

    if spx_rows or vix_rows:
        existing = load_json_safe(OHLC_FILE) or {"byDate": {}}
        by_date = existing.get("byDate", {})
        for row in spx_rows:
            d = row["date"]
            if d not in by_date: by_date[d] = {}
            # Open: solo se fija la primera vez (preserva el valor capturado al
            # abrir mercado; runs posteriores no lo tocan aunque Yahoo glitchee).
            new_open = r_int(row["open"])
            if by_date[d].get("spx_open") is None and new_open is not None:
                by_date[d]["spx_open"] = new_open
            # High/Low/Close: siempre se actualizan (evolucionan durante la sesión,
            # solo son definitivos tras 22:00 Madrid).
            if row["high"] is not None: by_date[d]["spx_high"] = r_int(row["high"])
            if row["low"] is not None: by_date[d]["spx_low"] = r_int(row["low"])
            if row["close"] is not None: by_date[d]["spx_close"] = r_int(row["close"])
        for row in vix_rows:
            d = row["date"]
            if d not in by_date: by_date[d] = {}
            new_vix_open = r_vix(row["open"])
            if by_date[d].get("vix_open") is None and new_vix_open is not None:
                by_date[d]["vix_open"] = new_vix_open
            if row["close"] is not None: by_date[d]["vix_close"] = r_vix(row["close"])
        save_json(OHLC_FILE, {"lastUpdated": now_iso, "byDate": by_date})
        print(f"  ✓ OHLC: {len(by_date)} fechas guardadas")

    # Cross-check: si la última vela de Yahoo no es de hoy, NYSE no operó hoy
    # (festivo no listado, fallo de feed, etc.) → no escribir cadena con fecha falsa.
    latest_spx = spx_rows[-1]["date"] if spx_rows else None
    if latest_spx and latest_spx != today:
        print(f"  ⏭  Última vela SPX es {latest_spx}, no {today}. NYSE parece cerrado → omito cadena.")
        print(f"[{datetime.now(timezone.utc).isoformat()}] Done.")
        return

    # Dedup: si la cadena de hoy ya está capturada (otro cron disparó antes), no la
    # sobrescribimos. Queremos preservar el primer snapshot — el más cercano a +30
    # min de apertura, antes de que cambien las primas.
    chain_file = os.path.join(CHAINS_DIR, f"{today}.json")
    if os.path.exists(chain_file):
        print(f"  ⏭  Cadena de hoy ya existe ({chain_file}). Omito CBOE para preservar snapshot original.")
        print(f"[{datetime.now(timezone.utc).isoformat()}] Done.")
        return

    # CBOE chain
    try:
        raw = http_get_json("https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json")
        d = raw.get("data", {})
        spot = d.get("current_price") or d.get("last") or d.get("price")
        options = [parse_option(o) for o in d.get("options", [])]
        options = [o for o in options if o is not None]
        print(f"  ✓ CBOE: spot={spot}, {len(options)} opciones")

        # Solo capturamos el vencimiento del propio día (0DTE).
        relevant = []
        for e in sorted(set(o["expiration"] for o in options)):
            ed = datetime.strptime(e, "%Y-%m-%d").date()
            dte = (ed - today_d).days
            if dte == 0: relevant.append((e, dte))

        chain_out = {"date": today, "capturedAt": now_iso, "spot": spot, "expirations": {}}
        for e, dte in relevant:
            chain_out["expirations"][e] = {"dte": dte,
                "strikes": merge_strikes(options, e, spot, max_pct=5.0)}
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
