#!/usr/bin/env python3
"""Build ML dataset bricks from the dashboard data lake."""
import json
import math
import os
import statistics
from datetime import datetime, timezone

from fetch_daily import DATA_DIR, load_json_safe, save_json

OHLC_FILE = os.path.join(DATA_DIR, "daily-ohlc.json")
SPOTGAMMA_FILE = os.path.join(DATA_DIR, "spotgamma-levels.json")
OUT_DIR = os.path.join(DATA_DIR, "ml-dataset")
OUT_UNDERLYING_FILE = os.path.join(OUT_DIR, "subyacente.json")
OUT_SPOTGAMMA_FILE = os.path.join(OUT_DIR, "spotgamma.json")
OUT_INDEX = os.path.join(DATA_DIR, "ml-dataset-index.json")
MODEL_VERSION = "ml-dataset-v2"


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


def open_zone(open_pct):
    value = safe_num(open_pct)
    if value is None:
        return "unknown"
    if value < 0:
        return "below_put_wall"
    if value < 10:
        return "low_tail_0_10"
    if value <= 20:
        return "low_adjust_10_20"
    if value <= 30:
        return "low_adjust_20_30"
    if value <= 80:
        return "middle_30_80"
    if value <= 90:
        return "high_adjust_80_90"
    if value <= 100:
        return "high_tail_90_100"
    return "above_call_wall"


def build_spotgamma_features(spotgamma, underlying_rows):
    by_date = spotgamma.get("byDate") or {}
    underlying_by_date = {row["date"]: row for row in underlying_rows}
    rows = []
    previous = None
    for date_str in sorted(by_date):
        raw = by_date.get(date_str) or {}
        underlying = underlying_by_date.get(date_str)
        open_ = safe_num(underlying.get("open")) if underlying else None
        call_wall = safe_num(raw.get("callWall"))
        put_wall = safe_num(raw.get("putWall"))
        gamma_flip = safe_num(raw.get("gammaFlip"))
        vol_trigger = safe_num(raw.get("volTrigger"))
        wall_range = call_wall - put_wall if call_wall is not None and put_wall is not None else None
        open_pct_walls = None
        if open_ is not None and wall_range and wall_range > 0:
            open_pct_walls = (open_ - put_wall) / wall_range * 100

        row = {
            "date": date_str,
            "hasUnderlying": underlying is not None,
            "open": open_,
            "callWall": call_wall,
            "putWall": put_wall,
            "gammaFlip": gamma_flip,
            "volTrigger": vol_trigger,
            "wallRange": wall_range,
            "openPctWalls": open_pct_walls,
            "openZone": open_zone(open_pct_walls),
            "openInsideWalls": open_pct_walls is not None and 0 <= open_pct_walls <= 100,
            "openOperableRange10_90": open_pct_walls is not None and 10 <= open_pct_walls <= 90,
            "openLowTail0_10": open_pct_walls is not None and 0 <= open_pct_walls < 10,
            "openLowAdjust10_20": open_pct_walls is not None and 10 <= open_pct_walls <= 20,
            "openLowAdjust20_30": open_pct_walls is not None and 20 < open_pct_walls <= 30,
            "openMiddle30_80": open_pct_walls is not None and 30 < open_pct_walls <= 80,
            "openHighAdjust80_90": open_pct_walls is not None and 80 < open_pct_walls <= 90,
            "openHighTail90_100": open_pct_walls is not None and 90 < open_pct_walls <= 100,
            "openOutsideWalls": open_pct_walls is not None and (open_pct_walls < 0 or open_pct_walls > 100),
            "distanceOpenCallWall": call_wall - open_ if call_wall is not None and open_ is not None else None,
            "distanceOpenPutWall": open_ - put_wall if put_wall is not None and open_ is not None else None,
            "distanceOpenGammaFlip": open_ - gamma_flip if gamma_flip is not None and open_ is not None else None,
            "distanceOpenVolTrigger": open_ - vol_trigger if vol_trigger is not None and open_ is not None else None,
            "callWallChange": call_wall - previous["callWall"] if previous and call_wall is not None and previous.get("callWall") is not None else None,
            "putWallChange": put_wall - previous["putWall"] if previous and put_wall is not None and previous.get("putWall") is not None else None,
            "gammaFlipChange": gamma_flip - previous["gammaFlip"] if previous and gamma_flip is not None and previous.get("gammaFlip") is not None else None,
            "volTriggerChange": vol_trigger - previous["volTrigger"] if previous and vol_trigger is not None and previous.get("volTrigger") is not None else None,
            "wallRangeChange": wall_range - previous["wallRange"] if previous and wall_range is not None and previous.get("wallRange") is not None else None,
            "source": raw.get("source") or "SpotGamma manual",
            "updatedAt": raw.get("updatedAt"),
        }
        rows.append(row)
        previous = row
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ohlc = load_json_safe(OHLC_FILE) or {"byDate": {}}
    spotgamma = load_json_safe(SPOTGAMMA_FILE) or {"byDate": {}}
    underlying_rows = build_underlying_features(ohlc)
    spotgamma_rows = build_spotgamma_features(spotgamma, underlying_rows)
    now = datetime.now(timezone.utc).isoformat()
    underlying_payload = {
        "version": MODEL_VERSION,
        "lastUpdated": now,
        "block": "subyacente",
        "description": "Primer bloque del dataset ML: features diarios del subyacente SPX y VIX derivados de daily-ohlc.json.",
        "source": "data/daily-ohlc.json",
        "rows": underlying_rows,
    }
    spotgamma_payload = {
        "version": MODEL_VERSION,
        "lastUpdated": now,
        "block": "spotgamma",
        "description": "Segundo bloque del dataset ML: niveles SpotGamma manuales alineados por fecha aplicada a la sesion.",
        "source": "data/spotgamma-levels.json",
        "rows": spotgamma_rows,
    }
    save_json(OUT_UNDERLYING_FILE, underlying_payload)
    save_json(OUT_SPOTGAMMA_FILE, spotgamma_payload)
    index = {
        "version": MODEL_VERSION,
        "lastUpdated": now,
        "blocks": {
            "subyacente": {
                "status": "available",
                "file": "data/ml-dataset/subyacente.json",
                "rows": len(underlying_rows),
                "firstDate": underlying_rows[0]["date"] if underlying_rows else None,
                "lastDate": underlying_rows[-1]["date"] if underlying_rows else None,
            },
            "spotgamma": {
                "status": "available",
                "file": "data/ml-dataset/spotgamma.json",
                "rows": len(spotgamma_rows),
                "firstDate": spotgamma_rows[0]["date"] if spotgamma_rows else None,
                "lastDate": spotgamma_rows[-1]["date"] if spotgamma_rows else None,
            },
            "gex": {"status": "pending"},
            "dex": {"status": "pending"},
            "vol_surface": {"status": "pending"},
            "premiums_labels": {"status": "pending"},
        },
    }
    save_json(OUT_INDEX, index)
    print(f"OK ML dataset: subyacente={len(underlying_rows)} rows, spotgamma={len(spotgamma_rows)} rows")


if __name__ == "__main__":
    main()
