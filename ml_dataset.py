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
GEX_INDEX_FILE = os.path.join(DATA_DIR, "gex-models-index.json")
GEX_MODELS_DIR = os.path.join(DATA_DIR, "gex-models")
DEX_INDEX_FILE = os.path.join(DATA_DIR, "dex-models-index.json")
DEX_MODELS_DIR = os.path.join(DATA_DIR, "dex-models")
OUT_DIR = os.path.join(DATA_DIR, "ml-dataset")
OUT_UNDERLYING_FILE = os.path.join(OUT_DIR, "subyacente.json")
OUT_SPOTGAMMA_FILE = os.path.join(OUT_DIR, "spotgamma.json")
OUT_GEX_FILE = os.path.join(OUT_DIR, "gex.json")
OUT_DEX_FILE = os.path.join(OUT_DIR, "dex.json")
OUT_INDEX = os.path.join(DATA_DIR, "ml-dataset-index.json")
MODEL_VERSION = "ml-dataset-v4"

GEX_MODEL_IDS = ("pure_gex", "liquidity_weighted", "spotgamma_fit")
GEX_MODE_IDS = ("next", "multi")
GEX_LEVEL_KEYS = ("callWall", "putWall", "gammaFlip", "volTrigger")
DEX_LEVEL_KEYS = ("callDeltaWall", "putDeltaWall", "dexFlip", "maxPositiveDex", "maxNegativeDex")


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


def median(values):
    clean = sorted(v for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2


def value_range(values):
    clean = [v for v in values if v is not None and math.isfinite(float(v))]
    return max(clean) - min(clean) if clean else None


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


def gex_sign(value):
    value = safe_num(value)
    if value is None:
        return "unknown"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "flat"


def nearest_distance(open_, levels):
    distances = []
    for key in GEX_LEVEL_KEYS:
        value = safe_num((levels or {}).get(key))
        if value is not None and open_ is not None:
            distances.append(abs(open_ - value))
    return min(distances) if distances else None


def expiration_summary(levels):
    expirations = (levels or {}).get("expirationsUsed") or []
    dtes = [safe_num(item.get("dte")) for item in expirations if safe_num(item.get("dte")) is not None]
    return {
        "expirationsUsedCount": len(expirations),
        "minDte": min(dtes) if dtes else None,
        "maxDte": max(dtes) if dtes else None,
        "firstExpiration": expirations[0].get("expiration") if expirations else None,
    }


def top_gex_summary(levels):
    top_rows = (levels or {}).get("topRows") or []
    histogram = (levels or {}).get("histogram") or []
    rows = top_rows if top_rows else histogram
    if not rows:
        return {
            "topAbsGexStrike": None,
            "topAbsGex": None,
            "topCallScoreStrike": None,
            "topPutScoreStrike": None,
        }
    top_abs = max(rows, key=lambda row: safe_num(row.get("absGex"), 0))
    top_call = max(rows, key=lambda row: safe_num(row.get("callScore"), 0))
    top_put = max(rows, key=lambda row: safe_num(row.get("putScore"), 0))
    return {
        "topAbsGexStrike": safe_num(top_abs.get("strike")),
        "topAbsGex": safe_num(top_abs.get("absGex")),
        "topCallScoreStrike": safe_num(top_call.get("strike")),
        "topPutScoreStrike": safe_num(top_put.get("strike")),
    }


def compact_gex_levels(levels, chain_spot, session_open):
    if not levels:
        return None
    call_wall = safe_num(levels.get("callWall"))
    put_wall = safe_num(levels.get("putWall"))
    gamma_flip = safe_num(levels.get("gammaFlip"))
    vol_trigger = safe_num(levels.get("volTrigger"))
    wall_range = call_wall - put_wall if call_wall is not None and put_wall is not None else None
    open_pct_walls = None
    if session_open is not None and wall_range and wall_range > 0:
        open_pct_walls = (session_open - put_wall) / wall_range * 100
    compact = {
        "callWall": call_wall,
        "putWall": put_wall,
        "gammaFlip": gamma_flip,
        "volTrigger": vol_trigger,
        "wallRange": wall_range,
        "netAtSpot": safe_num(levels.get("netAtSpot")),
        "netAtSpotSign": gex_sign(levels.get("netAtSpot")),
        "distanceChainSpotCallWall": call_wall - chain_spot if call_wall is not None and chain_spot is not None else None,
        "distanceChainSpotPutWall": chain_spot - put_wall if put_wall is not None and chain_spot is not None else None,
        "distanceChainSpotGammaFlip": chain_spot - gamma_flip if gamma_flip is not None and chain_spot is not None else None,
        "distanceChainSpotVolTrigger": chain_spot - vol_trigger if vol_trigger is not None and chain_spot is not None else None,
        "distanceOpenCallWall": call_wall - session_open if call_wall is not None and session_open is not None else None,
        "distanceOpenPutWall": session_open - put_wall if put_wall is not None and session_open is not None else None,
        "distanceOpenGammaFlip": session_open - gamma_flip if gamma_flip is not None and session_open is not None else None,
        "distanceOpenVolTrigger": session_open - vol_trigger if vol_trigger is not None and session_open is not None else None,
        "openPctGexWalls": open_pct_walls,
        "openZoneGexWalls": open_zone(open_pct_walls),
        "distanceOpenNearestGexLevel": nearest_distance(session_open, levels),
    }
    compact.update(expiration_summary(levels))
    compact.update(top_gex_summary(levels))
    return compact


def consensus_for_scope(models, scope):
    selected = []
    for model_id, by_mode in (models or {}).items():
        if scope == "all":
            selected.extend((by_mode.get(mode) for mode in GEX_MODE_IDS if by_mode.get(mode)))
        else:
            levels = by_mode.get(scope)
            if levels:
                selected.append(levels)
    out = {
        "scope": scope,
        "readings": len(selected),
        "positiveNetAtSpotCount": sum(1 for levels in selected if levels.get("netAtSpotSign") == "positive"),
        "negativeNetAtSpotCount": sum(1 for levels in selected if levels.get("netAtSpotSign") == "negative"),
    }
    for key in GEX_LEVEL_KEYS:
        values = [safe_num(levels.get(key)) for levels in selected]
        center = median(values)
        out[f"{key}Consensus"] = center
        out[f"{key}DispersionRange"] = value_range(values)
        out[f"{key}DispersionStdev"] = stdev(values)
        out[f"{key}AgreementWithin25Count"] = (
            sum(1 for value in values if value is not None and center is not None and abs(value - center) <= 25)
            if center is not None else 0
        )
    return out


def gex_pair_features(models):
    out = {}
    for model_id in GEX_MODEL_IDS:
        next_levels = (models.get(model_id) or {}).get("next") or {}
        multi_levels = (models.get(model_id) or {}).get("multi") or {}
        prefix = model_id
        for key in GEX_LEVEL_KEYS:
            next_value = safe_num(next_levels.get(key))
            multi_value = safe_num(multi_levels.get(key))
            out[f"{prefix}_{key}_nextMultiDiff"] = (
                multi_value - next_value if next_value is not None and multi_value is not None else None
            )
    for mode in GEX_MODE_IDS:
        pure = (models.get("pure_gex") or {}).get(mode) or {}
        liquidity = (models.get("liquidity_weighted") or {}).get(mode) or {}
        fit = (models.get("spotgamma_fit") or {}).get(mode) or {}
        for key in GEX_LEVEL_KEYS:
            pure_value = safe_num(pure.get(key))
            liquidity_value = safe_num(liquidity.get(key))
            fit_value = safe_num(fit.get(key))
            out[f"{mode}_{key}_pureLiquidityDiff"] = (
                liquidity_value - pure_value if pure_value is not None and liquidity_value is not None else None
            )
            out[f"{mode}_{key}_fitLiquidityDiff"] = (
                fit_value - liquidity_value if liquidity_value is not None and fit_value is not None else None
            )
    return out


def compact_gex_diagnostics(record):
    diagnostics = record.get("diagnostics") or {}
    out = {
        "referenceAvailable": bool(((record.get("reference") or {}).get("spotgamma"))),
        "byModel": {},
    }
    statuses = []
    errors = []
    for model_id in GEX_MODEL_IDS:
        out["byModel"][model_id] = {}
        for mode in GEX_MODE_IDS:
            diag = ((diagnostics.get(model_id) or {}).get(mode) or {})
            spotgamma_error = diag.get("spotgammaError") or {}
            operation = diag.get("operation") or {}
            weighted_error = safe_num(spotgamma_error.get("weightedError"))
            if weighted_error is not None:
                errors.append(weighted_error)
            status = operation.get("status")
            if status:
                statuses.append(status)
            out["byModel"][model_id][mode] = {
                "spotgammaWeightedError": weighted_error,
                "operationStatus": status,
                "openPct": safe_num(operation.get("openPct")),
                "sellCall": safe_num(operation.get("sellCall")),
                "sellPut": safe_num(operation.get("sellPut")),
                "upperTouch": operation.get("upperTouch"),
                "lowerTouch": operation.get("lowerTouch"),
                "adjustment": operation.get("adjustment"),
                "reason": operation.get("reason"),
            }
    out["avgSpotgammaWeightedError"] = mean(errors)
    out["bestSpotgammaWeightedError"] = min(errors) if errors else None
    out["winCount"] = statuses.count("win")
    out["lossCount"] = statuses.count("loss")
    out["noTradeCount"] = statuses.count("no_trade")
    out["pendingCount"] = statuses.count("pending")
    return out


def build_gex_features(gex_index, underlying_rows):
    dates = (gex_index or {}).get("dates") or []
    underlying_by_date = {row["date"]: row for row in underlying_rows}
    rows = []
    for source_date in sorted(dates):
        path = os.path.join(GEX_MODELS_DIR, f"{source_date}.json")
        record = load_json_safe(path)
        if not record:
            continue
        target_session = record.get("targetSession")
        underlying = underlying_by_date.get(target_session) or {}
        session_open = safe_num(underlying.get("open"))
        chain_spot = safe_num(record.get("spot"))
        compact_models = {}
        for model_id in GEX_MODEL_IDS:
            raw_model = (record.get("models") or {}).get(model_id) or {}
            compact_models[model_id] = {
                "name": raw_model.get("name"),
                "description": raw_model.get("description"),
            }
            for mode in GEX_MODE_IDS:
                compact_models[model_id][mode] = compact_gex_levels(raw_model.get(mode), chain_spot, session_open)

        row = {
            "date": target_session,
            "targetSession": target_session,
            "sourceChainDate": source_date,
            "sourceChain": record.get("sourceChain"),
            "gexModelVersion": record.get("version"),
            "generatedAt": record.get("generatedAt"),
            "capturedAt": record.get("capturedAt"),
            "chainSpot": chain_spot,
            "sessionOpen": session_open,
            "hasUnderlying": bool(underlying),
            "models": compact_models,
            "consensus": {
                "next": consensus_for_scope(compact_models, "next"),
                "multi": consensus_for_scope(compact_models, "multi"),
                "all": consensus_for_scope(compact_models, "all"),
            },
            "pairFeatures": gex_pair_features(compact_models),
            "diagnostics": compact_gex_diagnostics(record),
        }
        rows.append(row)
    rows.sort(key=lambda item: item.get("targetSession") or item.get("sourceChainDate") or "")
    return rows


def dex_top_summary(levels):
    top_rows = (levels or {}).get("topRows") or []
    histogram = (levels or {}).get("histogram") or []
    rows = top_rows if top_rows else histogram
    if not rows:
        return {
            "topAbsDexStrike": None,
            "topAbsDex": None,
            "topNetDexStrike": None,
            "topNegativeDexStrike": None,
        }
    top_abs = max(rows, key=lambda row: safe_num(row.get("absDex"), 0))
    top_net = max(rows, key=lambda row: safe_num(row.get("netDex"), 0))
    top_negative = min(rows, key=lambda row: safe_num(row.get("netDex"), 0))
    return {
        "topAbsDexStrike": safe_num(top_abs.get("strike")),
        "topAbsDex": safe_num(top_abs.get("absDex")),
        "topNetDexStrike": safe_num(top_net.get("strike")),
        "topNegativeDexStrike": safe_num(top_negative.get("strike")),
    }


def compact_dex_levels(levels, chain_spot, session_open):
    if not levels:
        return None
    call_wall = safe_num(levels.get("callDeltaWall"))
    put_wall = safe_num(levels.get("putDeltaWall"))
    dex_flip = safe_num(levels.get("dexFlip"))
    max_positive = safe_num(levels.get("maxPositiveDex"))
    max_negative = safe_num(levels.get("maxNegativeDex"))
    wall_range = call_wall - put_wall if call_wall is not None and put_wall is not None else None
    open_pct_walls = None
    if session_open is not None and wall_range and wall_range > 0:
        open_pct_walls = (session_open - put_wall) / wall_range * 100
    compact = {
        "callDeltaWall": call_wall,
        "putDeltaWall": put_wall,
        "dexFlip": dex_flip,
        "maxPositiveDex": max_positive,
        "maxNegativeDex": max_negative,
        "deltaWallRange": wall_range,
        "netDexAtSpot": safe_num(levels.get("netDexAtSpot")),
        "netDexAtSpotSign": gex_sign(levels.get("netDexAtSpot")),
        "distanceChainSpotCallDeltaWall": call_wall - chain_spot if call_wall is not None and chain_spot is not None else None,
        "distanceChainSpotPutDeltaWall": chain_spot - put_wall if put_wall is not None and chain_spot is not None else None,
        "distanceChainSpotDexFlip": chain_spot - dex_flip if dex_flip is not None and chain_spot is not None else None,
        "distanceChainSpotMaxPositiveDex": chain_spot - max_positive if max_positive is not None and chain_spot is not None else None,
        "distanceChainSpotMaxNegativeDex": chain_spot - max_negative if max_negative is not None and chain_spot is not None else None,
        "distanceOpenCallDeltaWall": call_wall - session_open if call_wall is not None and session_open is not None else None,
        "distanceOpenPutDeltaWall": session_open - put_wall if put_wall is not None and session_open is not None else None,
        "distanceOpenDexFlip": session_open - dex_flip if dex_flip is not None and session_open is not None else None,
        "distanceOpenMaxPositiveDex": session_open - max_positive if max_positive is not None and session_open is not None else None,
        "distanceOpenMaxNegativeDex": session_open - max_negative if max_negative is not None and session_open is not None else None,
        "openPctDexWalls": open_pct_walls,
        "openZoneDexWalls": open_zone(open_pct_walls),
    }
    compact.update(expiration_summary(levels))
    compact.update(dex_top_summary(levels))
    return compact


def build_dex_features(dex_index, underlying_rows):
    dates = (dex_index or {}).get("dates") or []
    underlying_by_date = {row["date"]: row for row in underlying_rows}
    rows = []
    previous = None
    for source_date in sorted(dates):
        path = os.path.join(DEX_MODELS_DIR, f"{source_date}.json")
        record = load_json_safe(path)
        if not record:
            continue
        target_session = record.get("targetSession")
        underlying = underlying_by_date.get(target_session) or {}
        session_open = safe_num(underlying.get("open"))
        chain_spot = safe_num(record.get("spot"))
        levels = compact_dex_levels(record.get("levels"), chain_spot, session_open)
        row = {
            "date": target_session,
            "targetSession": target_session,
            "sourceChainDate": source_date,
            "sourceChain": record.get("sourceChain"),
            "dexModelVersion": record.get("version"),
            "generatedAt": record.get("generatedAt"),
            "capturedAt": record.get("capturedAt"),
            "chainSpot": chain_spot,
            "sessionOpen": session_open,
            "hasUnderlying": bool(underlying),
            "model": record.get("model"),
            "levels": levels,
            "diagnostics": {
                "regime": (record.get("diagnostics") or {}).get("regime"),
            },
        }
        if previous and levels and previous.get("levels"):
            prev_levels = previous["levels"]
            row["changes"] = {
                "callDeltaWallChange": levels.get("callDeltaWall") - prev_levels.get("callDeltaWall") if levels.get("callDeltaWall") is not None and prev_levels.get("callDeltaWall") is not None else None,
                "putDeltaWallChange": levels.get("putDeltaWall") - prev_levels.get("putDeltaWall") if levels.get("putDeltaWall") is not None and prev_levels.get("putDeltaWall") is not None else None,
                "dexFlipChange": levels.get("dexFlip") - prev_levels.get("dexFlip") if levels.get("dexFlip") is not None and prev_levels.get("dexFlip") is not None else None,
                "netDexAtSpotChange": levels.get("netDexAtSpot") - prev_levels.get("netDexAtSpot") if levels.get("netDexAtSpot") is not None and prev_levels.get("netDexAtSpot") is not None else None,
            }
        else:
            row["changes"] = {
                "callDeltaWallChange": None,
                "putDeltaWallChange": None,
                "dexFlipChange": None,
                "netDexAtSpotChange": None,
            }
        rows.append(row)
        previous = row
    rows.sort(key=lambda item: item.get("targetSession") or item.get("sourceChainDate") or "")
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ohlc = load_json_safe(OHLC_FILE) or {"byDate": {}}
    spotgamma = load_json_safe(SPOTGAMMA_FILE) or {"byDate": {}}
    gex_index = load_json_safe(GEX_INDEX_FILE) or {"dates": []}
    dex_index = load_json_safe(DEX_INDEX_FILE) or {"dates": []}
    underlying_rows = build_underlying_features(ohlc)
    spotgamma_rows = build_spotgamma_features(spotgamma, underlying_rows)
    gex_rows = build_gex_features(gex_index, underlying_rows)
    dex_rows = build_dex_features(dex_index, underlying_rows)
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
    gex_payload = {
        "version": MODEL_VERSION,
        "lastUpdated": now,
        "block": "gex",
        "description": "Tercer bloque del dataset ML: firma compacta de los tres modelos GEX, consenso, dispersion y diagnosticos separados.",
        "source": "data/gex-models/*.json",
        "gexModelSourceVersion": gex_index.get("version"),
        "rows": gex_rows,
    }
    dex_payload = {
        "version": MODEL_VERSION,
        "lastUpdated": now,
        "block": "dex",
        "description": "Cuarto bloque del dataset ML: firma compacta del modelo Weighted DEX con delta walls, DEX flip, extremos y regimen.",
        "source": "data/dex-models/*.json",
        "dexModelSourceVersion": dex_index.get("version"),
        "rows": dex_rows,
    }
    save_json(OUT_UNDERLYING_FILE, underlying_payload)
    save_json(OUT_SPOTGAMMA_FILE, spotgamma_payload)
    save_json(OUT_GEX_FILE, gex_payload)
    save_json(OUT_DEX_FILE, dex_payload)
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
            "gex": {
                "status": "available",
                "file": "data/ml-dataset/gex.json",
                "rows": len(gex_rows),
                "firstDate": gex_rows[0]["targetSession"] if gex_rows else None,
                "lastDate": gex_rows[-1]["targetSession"] if gex_rows else None,
                "sourceVersion": gex_index.get("version"),
            },
            "dex": {
                "status": "available",
                "file": "data/ml-dataset/dex.json",
                "rows": len(dex_rows),
                "firstDate": dex_rows[0]["targetSession"] if dex_rows else None,
                "lastDate": dex_rows[-1]["targetSession"] if dex_rows else None,
                "sourceVersion": dex_index.get("version"),
            },
            "vol_surface": {"status": "pending"},
            "premiums_labels": {"status": "pending"},
        },
    }
    save_json(OUT_INDEX, index)
    print(f"OK ML dataset: subyacente={len(underlying_rows)} rows, spotgamma={len(spotgamma_rows)} rows, gex={len(gex_rows)} rows, dex={len(dex_rows)} rows")


if __name__ == "__main__":
    main()
