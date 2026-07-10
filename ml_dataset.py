#!/usr/bin/env python3
"""Build the first ML dataset brick: underlying/VIX daily features."""
import json
import math
import os
import statistics
from datetime import datetime, timezone

from fetch_daily import DATA_DIR, load_json_safe, save_json

OHLC_FILE = os.path.join(DATA_DIR, "daily-ohlc.json")
OUT_DIR = os.path.join(DATA_DIR, "ml-dataset")
OUT_FILE = os.path.join(OUT_DIR, "subyacente.json")
OUT_INDEX = os.path.join(DATA_DIR, "ml-dataset-index.json")
MODEL_VERSION = "ml-dataset-subyacente-v1"


def safe_num(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def pct(numerator, denominator):
    numerator = safe_num(numerator)
    denominator = safe_num(denominator)
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator * 100


def direction(value):
    value = safe_num(value)
    if value is None:
        return "unknown"
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def mean(values):
    clean = [v for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def stdev(values):
    clean = [v for v in values if v is not None and math.isfinite(float(v))]
    return statistics.stdev(clean) if len(clean) >= 2 else None


def rolling_sum(values):
    clean = [v for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) if clean else None


def day_name(date_str):
    from datetime import date

    y, m, d = [int(x) for x in date_str.split("-")]
    return date(y, m, d).strftime("%A").lower()


def complete_row(row):
    required = ("spx_open", "spx_high", "spx_low", "spx_close")
    return all(safe_num(row.get(key)) is not None for key in required)


def log_return(cur, prev):
    cur = safe_num(cur)
    prev = safe_num(prev)
    if cur is None or prev is None or cur <= 0 or prev <= 0:
        return None
    return math.log(cur / prev)


def build_underlying_features(ohlc):
    rows = []
    raw_by_date = ohlc.get("byDate") or {}
    for date_str in sorted(raw_by_date):
        row = raw_by_date.get(date_str) or {}
        if not complete_row(row):
            continue
        rows.append({
            "date": date_str,
            "open": safe_num(row.get("spx_open")),
            "high": safe_num(row.get("spx_high")),
            "low": safe_num(row.get("spx_low")),
            "close": safe_num(row.get("spx_close")),
            "vixOpen": safe_num(row.get("vix_open")),
            "vixClose": safe_num(row.get("vix_close")),
        })

    features = []
    close_returns = []
    ranges_pct = []
    for idx, row in enumerate(rows):
        prev = rows[idx - 1] if idx > 0 else None
        prev_close = prev["close"] if prev else None
        prev_high = prev["high"] if prev else None
        prev_low = prev["low"] if prev else None
        prev_vix_close = prev["vixClose"] if prev else None
        gap_abs = row["open"] - prev_close if prev_close is not None else None
        gap_pct = pct(gap_abs, prev_close) if gap_abs is not None else None
        prev_range = prev_high - prev_low if prev_high is not None and prev_low is not None else None
        open_position_prev_range = None
        if prev_range and prev_range > 0:
            open_position_prev_range = (row["open"] - prev_low) / prev_range * 100

        intraday_return_pct = pct(row["close"] - row["open"], row["open"])
        close_to_close_return = log_return(row["close"], prev_close)
        range_1d = row["high"] - row["low"]
        range_pct_1d = pct(range_1d, row["open"])
        close_returns.append(close_to_close_return)
        ranges_pct.append(range_pct_1d)

        trailing_returns_3 = close_returns[max(0, len(close_returns) - 3):]
        trailing_returns_5 = close_returns[max(0, len(close_returns) - 5):]
        trailing_ranges_3 = ranges_pct[max(0, len(ranges_pct) - 3):]
        trailing_ranges_5 = ranges_pct[max(0, len(ranges_pct) - 5):]
        trailing_returns_10 = close_returns[max(0, len(close_returns) - 10):]

        vix_change = row["vixClose"] - row["vixOpen"] if row["vixClose"] is not None and row["vixOpen"] is not None else None
        vix_gap = row["vixOpen"] - prev_vix_close if row["vixOpen"] is not None and prev_vix_close is not None else None
        day = day_name(row["date"])
        features.append({
            "date": row["date"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "prevClose": prev_close,
            "prevHigh": prev_high,
            "prevLow": prev_low,
            "gapAbs": gap_abs,
            "gapPct": gap_pct,
            "overnightDirection": direction(gap_abs),
            "intradayReturnPct": intraday_return_pct,
            "closeToCloseReturn": close_to_close_return,
            "openVsPrevHigh": row["open"] - prev_high if prev_high is not None else None,
            "openVsPrevLow": row["open"] - prev_low if prev_low is not None else None,
            "distanceOpenPrevHighPct": pct(prev_high - row["open"], row["open"]) if prev_high is not None else None,
            "distanceOpenPrevLowPct": pct(row["open"] - prev_low, row["open"]) if prev_low is not None else None,
            "openPositionPrevRange": open_position_prev_range,
            "range1d": range_1d,
            "rangePct1d": range_pct_1d,
            "rangePct3dAvg": mean(trailing_ranges_3),
            "rangePct5dAvg": mean(trailing_ranges_5),
            "realizedVol5d": stdev(trailing_returns_5),
            "realizedVol10d": stdev(trailing_returns_10),
            "trend3d": rolling_sum(trailing_returns_3),
            "trend5d": rolling_sum(trailing_returns_5),
            "trend3dDirection": direction(rolling_sum(trailing_returns_3)),
            "trend5dDirection": direction(rolling_sum(trailing_returns_5)),
            "vixOpen": row["vixOpen"],
            "vixClose": row["vixClose"],
            "prevVixClose": prev_vix_close,
            "vixChange": vix_change,
            "vixChangePct": pct(vix_change, row["vixOpen"]) if vix_change is not None else None,
            "vixGap": vix_gap,
            "vixGapPct": pct(vix_gap, prev_vix_close) if vix_gap is not None else None,
            "dayOfWeek": day,
            "isMonday": day == "monday",
            "isFriday": day == "friday",
        })
    return features


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ohlc = load_json_safe(OHLC_FILE) or {"byDate": {}}
    rows = build_underlying_features(ohlc)
    payload = {
        "version": MODEL_VERSION,
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "block": "subyacente",
        "description": "Primer bloque del dataset ML: features diarios del subyacente SPX y VIX derivados de daily-ohlc.json.",
        "source": "data/daily-ohlc.json",
        "rows": rows,
    }
    save_json(OUT_FILE, payload)
    index = {
        "version": MODEL_VERSION,
        "lastUpdated": payload["lastUpdated"],
        "blocks": {
            "subyacente": {
                "status": "available",
                "file": "data/ml-dataset/subyacente.json",
                "rows": len(rows),
                "firstDate": rows[0]["date"] if rows else None,
                "lastDate": rows[-1]["date"] if rows else None,
            },
            "spotgamma": {"status": "pending"},
            "gex": {"status": "pending"},
            "dex": {"status": "pending"},
            "vol_surface": {"status": "pending"},
            "premiums_labels": {"status": "pending"},
        },
    }
    save_json(OUT_INDEX, index)
    print(f"OK ML dataset subyacente: {len(rows)} rows")


if __name__ == "__main__":
    main()
