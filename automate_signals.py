#!/usr/bin/env python3
"""Generate SPX-IC signals, persist their state, and send idempotent email cards."""
import argparse
import html
import json
import math
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fetch_daily import DATA_DIR, load_json_safe, save_json
from monitor_premiums import (
    CONTRACT_MULTIPLIER,
    HISTORY_DIR,
    SPREAD_WIDTH,
    compute_levels,
    is_opex_day,
    no_trade_reason,
    previous_close_chain,
    select_legs,
)

SIGNALS_DIR = os.path.join(DATA_DIR, "signals")
SIGNALS_INDEX = os.path.join(DATA_DIR, "signals-index.json")
OHLC_FILE = os.path.join(DATA_DIR, "daily-ohlc.json")
MADRID = ZoneInfo("Europe/Madrid")

NAVY = "#123a60"
BLUE_LINE = "#c7d8e8"
BLUE_SOFT = "#eef5fb"
GOLD = "#c9a227"
GREEN = "#23824d"
RED = "#b73232"
MUTED = "#617585"


def money(value):
    sign = "+" if value > 0 else ""
    return f"{sign}${value:,.2f}"


def points(value):
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}"


def leg_label(name):
    labels = {
        "buy_call": "Compra Call",
        "sell_call": "Venta Call",
        "sell_put": "Venta Put",
        "buy_put": "Compra Put",
    }
    return labels[name]


def leg_entry_value(leg):
    return -float(leg["entryAsk"]) if leg["side"] == "buy" else float(leg["entryBid"])


def ordered_legs(signal):
    return [
        (name, signal["legs"][name])
        for name in ("buy_call", "sell_call", "sell_put", "buy_put")
        if name in signal.get("legs", {})
    ]


def base_card(title, eyebrow, message, color, content, footer):
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:620px;margin:auto;border:1px solid {BLUE_LINE};border-top:5px solid {color};border-radius:7px;overflow:hidden;background:#fff">
      <div style="background:{NAVY};color:#fff;padding:18px 20px">
        <div style="font-size:10px;text-transform:uppercase;color:#cbddea;font-weight:700">{html.escape(eyebrow)}</div>
        <h2 style="margin:5px 0 0;font-size:21px">{html.escape(title)}</h2>
      </div>
      <div style="padding:18px 20px;color:#172b3a">
        <div style="padding:10px 12px;border-radius:5px;margin-bottom:14px;font-size:12px;font-weight:700;background:{color}18;border:1px solid {color};color:{color}">
          {html.escape(message)}
        </div>
        {content}
        <div style="margin-top:15px;color:{MUTED};font-size:10px;text-align:center">{html.escape(footer)}</div>
      </div>
    </div>"""


def legs_table(signal):
    rows = []
    for name, leg in ordered_legs(signal):
        value = leg_entry_value(leg)
        rows.append(f"""
          <tr>
            <td style="padding:8px 5px;border-bottom:1px solid #e1eaf1">{leg_label(name)}</td>
            <td style="padding:8px 5px;border-bottom:1px solid #e1eaf1;text-align:right">{leg['strike']:,.0f}</td>
            <td style="padding:8px 5px;border-bottom:1px solid #e1eaf1;text-align:right;font-weight:700">{points(value)}</td>
          </tr>""")
    return f"""
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <tr style="color:{MUTED};font-size:10px;text-transform:uppercase">
          <th style="padding:5px;text-align:left">Posición</th><th style="padding:5px;text-align:right">Strike</th><th style="padding:5px;text-align:right">Prima</th>
        </tr>
        {''.join(rows)}
      </table>"""


def entry_card(signal):
    date = signal["date"]
    if signal["status"] == "no_trade":
        content = f"""
          <div style="text-align:center;padding:18px 8px 12px">
            <div style="font-size:42px;line-height:1">🚫</div>
            <div style="margin-top:12px;font-weight:800;color:{RED}">Sin operación de venta de volatilidad para hoy</div>
          </div>"""
        return (
            f"SPX-IC · NO OPERAR · {date}",
            base_card(
                "🚫 NO OPERAR · Venta de volatilidad",
                f"SPX-IC · Señal diaria · {date} · 16:07 Madrid",
                "La sesión queda excluida por los filtros de riesgo del modelo.",
                RED,
                content,
                "No se abrirá operación y no se enviará resumen nocturno de sesión.",
            ),
        )

    capital = signal.get("capitalPercent", 100)
    opex = signal.get("opex", False)
    message = "La sesión supera los filtros de entrada. Ejecutar los niveles indicados."
    if capital == 50:
        message += " Operar únicamente con el 50% del capital destinado."
    if opex:
        message += " Sesión OPEX: ejecutar solamente el call spread."
    content = legs_table(signal) + f"""
      <div style="margin-top:12px;padding-top:10px;border-top:2px solid {GOLD};display:flex;justify-content:space-between;font-weight:800">
        <span>Crédito neto estimado</span><span>{points(signal['entryCredit'])}</span>
      </div>"""
    return (
        f"SPX-IC · OPERAR · {date}",
        base_card(
            "OPERAR · Call Spread OPEX 0DTE" if opex else "OPERAR · Iron Condor 0DTE",
            f"SPX-IC · Señal diaria · {date} · 16:07 Madrid",
            message,
            GREEN,
            content,
            "Precios de prima extraídos aproximadamente a las 16:07 hora Madrid.",
        ),
    )


def daily_result_section(signal):
    result = signal["result"]
    color = GREEN if result["pnl"] >= 0 else RED
    return legs_table(signal) + f"""
      <div style="margin-top:12px;padding-top:10px;border-top:2px solid {GOLD};display:flex;justify-content:space-between;font-weight:800">
        <span>Resultado estimado</span><span style="color:{color}">{money(result['pnl'])}</span>
      </div>
      <div style="font-size:10px;color:{MUTED};text-align:right;margin-top:4px">Por un contrato de IC ejercido</div>"""


def week_dates(date_string):
    date = datetime.strptime(date_string, "%Y-%m-%d").date()
    monday = date - timedelta(days=date.weekday())
    return [(monday + timedelta(days=i)).isoformat() for i in range(5)]


def weekly_section(date_string):
    rows = []
    total = 0
    names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    for name, date in zip(names, week_dates(date_string)):
        signal = load_json_safe(os.path.join(SIGNALS_DIR, f"{date}.json"))
        if not signal:
            label, amount, bg, color = "Sin registro", "—", "#fff", MUTED
        elif signal.get("status") == "no_trade":
            label, amount, bg, color = "No operado", "$0.00", "#fff", MUTED
        elif signal.get("result"):
            pnl = float(signal["result"]["pnl"])
            total += pnl
            good = pnl >= 0
            label, amount = ("Operado · Ganancia" if good else "Operado · Pérdida"), money(pnl)
            bg, color = ("#eaf6ee", GREEN) if good else ("#fdeaea", RED)
        else:
            label, amount, bg, color = "Operación pendiente", "—", "#fff8df", GOLD
        rows.append(f"""
          <tr style="background:{bg}">
            <td style="padding:8px 6px;border-bottom:1px solid #e1eaf1"><b>{name}</b><br><span style="font-size:9px;color:{MUTED}">{date}</span></td>
            <td style="padding:8px 6px;border-bottom:1px solid #e1eaf1">{label}</td>
            <td style="padding:8px 6px;border-bottom:1px solid #e1eaf1;text-align:right;font-weight:700;color:{color}">{amount}</td>
          </tr>""")
    total_color = GREEN if total >= 0 else RED
    return f"""
      <table style="width:100%;border-collapse:collapse;font-size:12px">{''.join(rows)}</table>
      <div style="margin-top:12px;padding-top:10px;border-top:2px solid {GOLD};display:flex;justify-content:space-between;font-weight:800">
        <span>Balance semanal</span><span style="color:{total_color}">{money(total)}</span>
      </div>"""


def close_card(signal, include_weekly):
    date = signal["date"]
    sections = []
    title = "RESUMEN DEL DÍA · Operación finalizada"
    message = "Resumen de las primas seleccionadas y del resultado estimado."
    result = signal.get("result")
    color = GREEN if result and result["pnl"] >= 0 else RED
    if signal.get("status") == "active":
        sections.append(daily_result_section(signal))
    if include_weekly:
        if sections:
            sections.append(f'<div style="height:1px;background:{BLUE_LINE};margin:20px 0"></div>')
        sections.append("<h3 style='color:#123a60;margin:0 0 10px'>Resumen semanal</h3>" + weekly_section(date))
        title = "RESUMEN DEL DÍA + SEMANA · SPX-IC" if signal.get("status") == "active" else "RESUMEN SEMANAL · SPX-IC"
        message = "Balance agregado de las sesiones de esta semana."
        color = GOLD
    return (
        f"SPX-IC · {'Resultado + semana' if include_weekly and signal.get('status') == 'active' else 'Resumen semanal' if include_weekly else 'Resultado'} · {date}",
        base_card(
            title,
            f"SPX-IC · Cierre · {date} · 22:32 Madrid",
            message,
            color,
            "".join(sections),
            "Resultado estimado por un contrato. Las pérdidas pueden variar si la posición se cierra antes del vencimiento.",
        ),
    )


def load_signal(date):
    return load_json_safe(os.path.join(SIGNALS_DIR, f"{date}.json"))


def save_signal(signal):
    os.makedirs(SIGNALS_DIR, exist_ok=True)
    save_json(os.path.join(SIGNALS_DIR, f"{signal['date']}.json"), signal)
    index = load_json_safe(SIGNALS_INDEX) or {"dates": []}
    if signal["date"] not in index["dates"]:
        index["dates"].append(signal["date"])
        index["dates"].sort()
    index["lastUpdated"] = datetime.now(timezone.utc).isoformat()
    save_json(SIGNALS_INDEX, index)


def previous_failed(date):
    index = load_json_safe(SIGNALS_INDEX) or {"dates": []}
    for previous_date in sorted((item for item in index.get("dates", []) if item < date), reverse=True):
        signal = load_signal(previous_date)
        if signal and signal.get("status") == "active" and signal.get("result"):
            return float(signal["result"]["pnl"]) < 0
    return False


def create_entry_signal(date):
    source_date, close_chain = previous_close_chain(date)
    levels = compute_levels(close_chain or {})
    if not levels:
        raise RuntimeError("no se pudieron calcular los niveles del cierre anterior")
    reason = no_trade_reason(date, levels)
    signal = {
        "date": date,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceCloseDate": source_date,
        "emails": {},
    }
    if reason:
        signal.update({"status": "no_trade", "reason": reason})
        return signal
    _, legs, entry_credit = select_legs(date, levels)
    signal.update({
        "status": "active",
        "opex": is_opex_day(date),
        "capitalPercent": 50 if previous_failed(date) else 100,
        "spreadWidth": SPREAD_WIDTH,
        "entryCredit": entry_credit,
        "legs": legs,
    })
    return signal


def result_for(signal):
    history = load_json_safe(os.path.join(HISTORY_DIR, f"{signal['date']}.json"))
    if history and history.get("snapshots"):
        last = history["snapshots"][-1]
        return {
            "source": "premium_monitor",
            "capturedAt": last.get("capturedAt"),
            "pnl": float(last["pnl"]),
            "closeCost": float(last["closeCost"]),
            "multiple": last.get("multiple"),
        }
    ohlc = (load_json_safe(OHLC_FILE) or {}).get("byDate", {}).get(signal["date"], {})
    high, low = ohlc.get("spx_high"), ohlc.get("spx_low")
    sell_call = signal["legs"].get("sell_call")
    sell_put = signal["legs"].get("sell_put")
    touched = (
        sell_call and high is not None and float(high) >= float(sell_call["strike"])
    ) or (
        sell_put and low is not None and float(low) <= float(sell_put["strike"])
    )
    credit = float(signal["entryCredit"])
    pnl = (-(SPREAD_WIDTH - credit) if touched else credit) * CONTRACT_MULTIPLIER
    return {"source": "expiry_estimate", "pnl": pnl, "touched": bool(touched)}


def send_email(subject, card_html, idempotency_key, dry_run):
    if dry_run:
        print(f"  DRY RUN email: {subject} [{idempotency_key}]")
        return "dry-run"
    api_key = os.environ.get("RESEND_API_KEY")
    email_to = os.environ.get("ALERT_EMAIL_TO")
    if not api_key or not email_to:
        raise RuntimeError("faltan RESEND_API_KEY o ALERT_EMAIL_TO")
    payload = {
        "from": os.environ.get("ALERT_EMAIL_FROM") or "SPX-IC Dashboard <onboarding@resend.dev>",
        "to": [email_to],
        "subject": subject,
        "html": card_html,
    }
    request = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SPX-IC-GitHub-Actions/2.0",
            "Idempotency-Key": idempotency_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8")).get("id")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Resend HTTP {error.code}: {body}") from error


def run_entry(date, dry_run):
    signal = load_signal(date)
    if not signal:
        try:
            signal = create_entry_signal(date)
        except RuntimeError as exc:
            if "cadena de entrada" in str(exc):
                print(f"  SKIP: {exc}. Se reintentará en la ejecución de entrada.")
                return
            raise
    if signal.get("emails", {}).get("entry"):
        print("  SKIP: email de entrada ya registrado.")
        return
    subject, card = entry_card(signal)
    email_id = send_email(subject, card, f"spx-ic-entry-{date}", dry_run)
    signal.setdefault("emails", {})["entry"] = {"id": email_id, "sentAt": datetime.now(timezone.utc).isoformat()}
    save_signal(signal)
    print(f"  OK entrada: {signal['status']}")


def run_close(date, dry_run):
    signal = load_signal(date)
    if not signal:
        print("  SKIP: no existe señal de entrada.")
        return
    friday = datetime.strptime(date, "%Y-%m-%d").weekday() == 4
    email_key = "weekly" if friday else "result"
    if signal.get("emails", {}).get(email_key):
        print(f"  SKIP: email {email_key} ya registrado.")
        return
    if signal.get("status") == "no_trade" and not friday:
        print("  SKIP: día no operable y no es viernes.")
        return
    if signal.get("status") == "active":
        signal["result"] = result_for(signal)
    subject, card = close_card(signal, friday)
    email_id = send_email(subject, card, f"spx-ic-{email_key}-{date}", dry_run)
    signal.setdefault("emails", {})[email_key] = {"id": email_id, "sentAt": datetime.now(timezone.utc).isoformat()}
    save_signal(signal)
    print(f"  OK cierre: {email_key}")


def inferred_phase(now_madrid):
    return "close" if now_madrid.hour >= 22 else "entry"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("entry", "close"))
    parser.add_argument("--date", help="YYYY-MM-DD; por defecto fecha Madrid actual")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    now_madrid = datetime.now(MADRID)
    date = args.date or now_madrid.date().isoformat()
    if not args.phase and now_madrid.hour < 16:
        print(f"SPX-IC automation: Madrid {now_madrid:%H:%M}; todavía no es hora de señal.")
        return
    phase = args.phase or inferred_phase(now_madrid)
    dry_run = args.dry_run or os.environ.get("SPX_IC_DRY_RUN") == "1"
    print(f"SPX-IC automation: date={date}, phase={phase}, dry_run={dry_run}")
    if phase == "entry":
        run_entry(date, dry_run)
    else:
        run_close(date, dry_run)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERR automation: {exc}", file=sys.stderr)
        raise
