#!/usr/bin/env python3
"""Generate equity curve PDF for Benchmark and ML Shadow."""
import json
import math
import os
import shutil
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output", "pdf")
TMP_DIR = os.path.join(ROOT, "tmp", "pdfs")
REPORT_FILE = os.path.join(
    OUT_DIR,
    "spx_ic_curvas_balance_benchmark_ml_2026-06-23_2026-07-29.pdf",
)


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def fmt_money(value):
    value = num(value)
    if value is None:
        return "-"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(round(value)):,}".replace(",", ".")


def chain_strikes(chain, date):
    expirations = chain.get("expirations") or {}
    expiry = expirations.get(date)
    if expiry is None and expirations:
        expiry = next(iter(expirations.values()))
    return (expiry or {}).get("strikes") or []


def quote(chain, date, side, strike):
    strike = num(strike)
    if strike is None:
        return None
    prefix = "call" if side == "call" else "put"
    for row in chain_strikes(chain, date):
        if num(row.get("strike")) == strike:
            return {
                "bid": num(row.get(f"{prefix}_bid")),
                "ask": num(row.get(f"{prefix}_ask")),
            }
    return None


def ml_shadow_close_pnl(date):
    shadow = load_json(os.path.join(DATA_DIR, "ml-shadow", "predictions", f"{date}.json"), {})
    decision = shadow.get("decision") or {}
    if decision.get("action") == "no_operar":
        return 0.0

    entry_chain = load_json(os.path.join(DATA_DIR, "chains", f"{date}.json"))
    close_chain = load_json(os.path.join(DATA_DIR, "chains-close", f"{date}.json"))
    if not entry_chain or not close_chain:
        return 0.0

    total = 0.0
    for wing in decision.get("selectedWings") or []:
        side = wing.get("side")
        sell_entry = quote(entry_chain, date, side, wing.get("strike"))
        buy_entry = quote(entry_chain, date, side, wing.get("buyStrike"))
        sell_close = quote(close_chain, date, side, wing.get("strike"))
        buy_close = quote(close_chain, date, side, wing.get("buyStrike"))
        if not (sell_entry and buy_entry and sell_close and buy_close):
            continue
        if None in (sell_entry["bid"], buy_entry["ask"], sell_close["ask"], buy_close["bid"]):
            continue

        entry_credit = (sell_entry["bid"] - buy_entry["ask"]) * 100
        close_cost = max(0.0, (sell_close["ask"] - buy_close["bid"]) * 100)
        total += entry_credit - close_cost

    return round(total, 2)


def build_series():
    index = load_json(os.path.join(DATA_DIR, "ml-shadow", "index.json"), {})
    dates = sorted(index.get("dates") or [])

    benchmark_daily = []
    shadow_daily = []
    benchmark_equity = []
    shadow_equity = []
    benchmark_balance = 0.0
    shadow_balance = 0.0

    for date in dates:
        signal = load_json(os.path.join(DATA_DIR, "signals", f"{date}.json"), {})
        benchmark_pnl = 0.0
        if signal.get("status") == "active":
            benchmark_pnl = num((signal.get("result") or {}).get("pnl")) or 0.0

        shadow_pnl = ml_shadow_close_pnl(date)
        benchmark_balance += benchmark_pnl
        shadow_balance += shadow_pnl

        benchmark_daily.append(round(benchmark_pnl, 2))
        shadow_daily.append(round(shadow_pnl, 2))
        benchmark_equity.append(round(benchmark_balance, 2))
        shadow_equity.append(round(shadow_balance, 2))

    return {
        "dates": dates,
        "benchmark": {"daily": benchmark_daily, "equity": benchmark_equity},
        "shadow": {"daily": shadow_daily, "equity": shadow_equity},
    }


def get_font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def nice_ticks(min_value, max_value, count=5):
    if min_value == max_value:
        return [min_value]
    raw_step = (max_value - min_value) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(abs(raw_step)))
    residual = raw_step / magnitude
    if residual <= 1:
        step = magnitude
    elif residual <= 2:
        step = 2 * magnitude
    elif residual <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    start = math.floor(min_value / step) * step
    end = math.ceil(max_value / step) * step
    ticks = []
    value = start
    while value <= end + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def draw_chart(path, title, dates, daily, equity, subtitle):
    width, height = 1800, 780
    margin_l, margin_r, margin_t, margin_b = 145, 70, 155, 105
    plot_l, plot_t = margin_l, margin_t
    plot_r, plot_b = width - margin_r, height - margin_b
    plot_w, plot_h = plot_r - plot_l, plot_b - plot_t

    img = Image.new("RGB", (width, height), "#f4f4f1")
    draw = ImageDraw.Draw(img)
    font_title = get_font(44, bold=True)
    font_subtitle = get_font(25)
    font_axis = get_font(22)
    font_small = get_font(20)
    font_bold = get_font(23, bold=True)

    draw.rounded_rectangle((25, 25, width - 25, height - 25), radius=36, fill="#ffffff", outline="#deded8", width=2)
    draw.text((65, 50), title, fill="#111111", font=font_title)
    draw.text((68, 104), subtitle, fill="#626262", font=font_subtitle)

    all_values = list(daily) + list(equity) + [0]
    y_min = min(all_values)
    y_max = max(all_values)
    span = y_max - y_min if y_max != y_min else 1
    pad = max(80, span * 0.12)
    y_min -= pad
    y_max += pad

    def ymap(value):
        return plot_b - ((value - y_min) / (y_max - y_min)) * plot_h

    def xmap(index):
        if len(dates) <= 1:
            return plot_l + plot_w / 2
        return plot_l + index * plot_w / (len(dates) - 1)

    ticks = nice_ticks(y_min, y_max)
    if len(ticks) >= 2:
        y_min, y_max = ticks[0], ticks[-1]

    for tick in ticks:
        y = ymap(tick)
        color = "#b8b8b8" if abs(tick) < 1e-9 else "#dddddd"
        draw.line((plot_l, y, plot_r, y), fill=color, width=2 if abs(tick) < 1e-9 else 1)
        draw.text((35, y - 13), fmt_money(tick), fill="#595959", font=font_small)

    draw.line((plot_l, plot_b, plot_r, plot_b), fill="#111111", width=2)
    draw.line((plot_l, plot_t, plot_l, plot_b), fill="#111111", width=2)

    zero_y = ymap(0)
    slot = plot_w / max(1, len(dates))
    bar_w = max(10, min(38, slot * 0.55))
    for i, value in enumerate(daily):
        x = xmap(i)
        y = ymap(value)
        color = "#0fa36b" if value >= 0 else "#d94848"
        top = min(y, zero_y)
        bottom = max(y, zero_y)
        if abs(bottom - top) < 2:
            bottom = top + 2
        draw.rounded_rectangle((x - bar_w / 2, top, x + bar_w / 2, bottom), radius=5, fill=color)

    points = [(xmap(i), ymap(v)) for i, v in enumerate(equity)]
    if len(points) > 1:
        draw.line(points, fill="#111111", width=7, joint="curve")
        draw.line(points, fill="#f7f7f7", width=3, joint="curve")
    for x, y in points:
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill="#111111", outline="#f7f7f7", width=3)

    last_x, last_y = points[-1]
    draw.rounded_rectangle((last_x - 110, last_y - 60, last_x + 110, last_y - 18), radius=18, fill="#111111")
    draw.text((last_x - 83, last_y - 54), fmt_money(equity[-1]), fill="#ffffff", font=font_bold)

    label_step = max(1, math.ceil(len(dates) / 8))
    for i, date in enumerate(dates):
        if i % label_step != 0 and i != len(dates) - 1:
            continue
        x = xmap(i)
        label = date[5:].replace("-", "/")
        draw.text((x - 33, plot_b + 24), label, fill="#4f4f4f", font=font_small)

    img.save(path, quality=95)


def make_pdf(series):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    benchmark_chart = os.path.join(TMP_DIR, "benchmark_equity.png")
    shadow_chart = os.path.join(TMP_DIR, "ml_shadow_equity.png")
    dates = series["dates"]

    benchmark_total = series["benchmark"]["equity"][-1] if dates else 0
    shadow_total = series["shadow"]["equity"][-1] if dates else 0
    benchmark_ops = sum(1 for v in series["benchmark"]["daily"] if abs(v) > 1e-9)
    shadow_ops = sum(1 for v in series["shadow"]["daily"] if abs(v) > 1e-9)
    shadow_losses = sum(1 for v in series["shadow"]["daily"] if v < 0)

    period = f"{dates[0]} a {dates[-1]}" if dates else "-"
    draw_chart(
        benchmark_chart,
        "Balance Benchmark SpotGamma",
        dates,
        series["benchmark"]["daily"],
        series["benchmark"]["equity"],
        f"Periodo {period} | barras: P&L diario | linea: balance acumulado | operaciones: {benchmark_ops} | resultado {fmt_money(benchmark_total)}",
    )
    draw_chart(
        shadow_chart,
        "Balance ML Shadow",
        dates,
        series["shadow"]["daily"],
        series["shadow"]["equity"],
        f"Periodo {period} | barras: P&L diario | linea: balance acumulado | operaciones: {shadow_ops} | perdidas: {shadow_losses} | resultado {fmt_money(shadow_total)}",
    )

    pdf = canvas.Canvas(REPORT_FILE, pagesize=A4)
    page_w, page_h = A4
    pdf.setTitle("SPX-IC Curvas de Balance Benchmark y ML Shadow")
    pdf.setAuthor("Codex / E. M. C")

    pdf.setFillColor(colors.HexColor("#f0f0ed"))
    pdf.rect(0, 0, page_w, page_h, stroke=0, fill=1)
    pdf.setFillColor(colors.HexColor("#111111"))
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawCentredString(page_w / 2, page_h - 22 * mm, "SPX-IC - Curvas de Balance")
    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(colors.HexColor("#555555"))
    pdf.drawCentredString(
        page_w / 2,
        page_h - 29 * mm,
        f"Benchmark vs ML Shadow | {period} | generado {datetime.now():%Y-%m-%d %H:%M}",
    )

    chart_w = page_w - 24 * mm
    chart_h = 94 * mm
    pdf.drawImage(benchmark_chart, 12 * mm, page_h - 132 * mm, width=chart_w, height=chart_h)
    pdf.drawImage(shadow_chart, 12 * mm, page_h - 235 * mm, width=chart_w, height=chart_h)

    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(colors.HexColor("#555555"))
    note = "Nota: Benchmark usa P&L real. ML Shadow usa P&L teorico al cierre desde cadenas entry/close."
    pdf.drawString(14 * mm, 13 * mm, note)
    pdf.save()

    return REPORT_FILE


def main():
    series = build_series()
    report = make_pdf(series)
    print(report)


if __name__ == "__main__":
    main()
