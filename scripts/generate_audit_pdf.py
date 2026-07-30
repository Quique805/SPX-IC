#!/usr/bin/env python3
"""Generate a PDF audit report for Benchmark vs ML Shadow."""
import json
import math
import os
from collections import defaultdict
from datetime import date, datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output", "pdf")
REPORT_FILE = os.path.join(
    OUT_DIR,
    "spx_ic_auditoria_benchmark_vs_ml_2026-06-23_2026-07-29.pdf",
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


def money(value):
    value = num(value)
    return None if value is None else round(value)


def fmt(value, digits=0, dash="-"):
    value = num(value)
    if value is None:
        return dash
    if digits:
        return f"{value:.{digits}f}"
    return f"{round(value):,}".replace(",", ".")


def fmt_money(value):
    value = num(value)
    if value is None:
        return "-"
    return "$" + f"{round(value):,}".replace(",", ".")


def fmt_pct(value, digits=2):
    value = num(value)
    if value is None:
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def week_key(date_str):
    y, m, d = [int(x) for x in date_str.split("-")]
    iso = date(y, m, d).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def close_band_metrics(close, call_sell=None, put_sell=None):
    close = num(close)
    call_sell = num(call_sell)
    put_sell = num(put_sell)
    if close is None or (call_sell is None and put_sell is None):
        return None, None, None
    if call_sell is not None and put_sell is not None:
        if put_sell <= close <= call_sell:
            dist_call = (call_sell - close) / call_sell * 100
            dist_put = (close - put_sell) / put_sell * 100
            if abs(dist_call) <= abs(dist_put):
                return True, "C", dist_call
            return True, "P", dist_put
        if close > call_sell:
            return False, "C", -(close - call_sell) / call_sell * 100
        return False, "P", -(put_sell - close) / put_sell * 100
    if call_sell is not None:
        if close <= call_sell:
            return True, "C", (call_sell - close) / call_sell * 100
        return False, "C", -(close - call_sell) / call_sell * 100
    if close >= put_sell:
        return True, "P", (close - put_sell) / put_sell * 100
    return False, "P", -(put_sell - close) / put_sell * 100


def leg_text(sell, buy, width=False):
    if sell is None:
        return "-"
    if buy is None:
        return fmt(sell)
    text = f"{fmt(sell)}/{fmt(buy)}"
    if width:
        text += f" {fmt(abs(num(buy) - num(sell)))}p"
    return text


def build_rows():
    ohlc = (load_json(os.path.join(DATA_DIR, "daily-ohlc.json"), {}) or {}).get("byDate", {})
    ml_index = load_json(os.path.join(DATA_DIR, "ml-shadow", "index.json"), {}) or {}
    ml_dates = ml_index.get("dates") or []

    benchmark_rows = []
    ml_rows = []

    for session in ml_dates:
        day = ohlc.get(session, {})
        high = num(day.get("spx_high"))
        low = num(day.get("spx_low"))
        close = num(day.get("spx_close"))

        signal = load_json(os.path.join(DATA_DIR, "signals", f"{session}.json"), {}) or {}
        premium_history = load_json(os.path.join(DATA_DIR, "premium-history", f"{session}.json"), {}) or {}
        open_wall = signal.get("openWall") or {}
        legs = signal.get("legs") or {}
        sell_call = legs.get("sell_call") or {}
        buy_call = legs.get("buy_call") or {}
        sell_put = legs.get("sell_put") or {}
        buy_put = legs.get("buy_put") or {}

        call_sell = num(sell_call.get("strike"))
        put_sell = num(sell_put.get("strike"))
        call_buy = num(buy_call.get("strike"))
        put_buy = num(buy_put.get("strike"))
        call_touch = high is not None and call_sell is not None and high >= call_sell
        put_touch = low is not None and put_sell is not None and low <= put_sell
        touch = (call_touch or put_touch) if signal.get("status") == "active" else None
        inside, nearest, close_pct = close_band_metrics(close, call_sell, put_sell)
        snapshots = premium_history.get("snapshots") or []
        multiples = [num(s.get("multiple")) for s in snapshots if num(s.get("multiple")) is not None]
        min_pnl = [num(s.get("pnl")) for s in snapshots if num(s.get("pnl")) is not None]
        skipped = ",".join((item.get("wing") or "") for item in (open_wall.get("skipped_wings") or [])) or "-"
        credit = num(signal.get("entryCredit"))

        benchmark_rows.append(
            {
                "date": session,
                "week": week_key(session),
                "status": signal.get("status") or "missing",
                "cw": num(open_wall.get("call_wall")),
                "pw": num(open_wall.get("put_wall")),
                "call": leg_text(call_sell, call_buy),
                "put": leg_text(put_sell, put_buy),
                "credit": money(credit * 100 if credit is not None else None),
                "touch": touch,
                "touch_side": ",".join(x for x, touched in (("C", call_touch), ("P", put_touch)) if touched) or "-",
                "close": close,
                "inside": inside,
                "nearest": nearest,
                "close_pct": close_pct,
                "pnl": money((signal.get("result") or {}).get("pnl")),
                "max_mult": max(multiples) if multiples else None,
                "min_pnl": money(min(min_pnl) if min_pnl else None),
                "skipped": skipped,
            }
        )

        shadow = load_json(os.path.join(DATA_DIR, "ml-shadow", "predictions", f"{session}.json"), {}) or {}
        decision = shadow.get("decision") or {}
        evaluation = shadow.get("evaluation") or {}
        context = shadow.get("context") or {}
        wings = decision.get("selectedWings") or []
        call = next((w for w in wings if w.get("side") == "call"), None)
        put = next((w for w in wings if w.get("side") == "put"), None)
        ml_call_sell = num(call.get("strike")) if call else None
        ml_put_sell = num(put.get("strike")) if put else None
        ml_call_buy = num(call.get("buyStrike")) if call else None
        ml_put_buy = num(put.get("buyStrike")) if put else None
        ml_inside, ml_nearest, ml_close_pct = close_band_metrics(context.get("close"), ml_call_sell, ml_put_sell)
        ml_touch = None
        if evaluation.get("available") and decision.get("action") != "no_operar":
            ml_touch = not bool(evaluation.get("noTouchAll"))
        profiles = "/".join((w.get("protectionProfile") or "-") for w in wings) or "-"
        ml_rows.append(
            {
                "date": session,
                "week": week_key(session),
                "action": decision.get("action") or "-",
                "call": leg_text(ml_call_sell, ml_call_buy, width=True),
                "put": leg_text(ml_put_sell, ml_put_buy, width=True),
                "credit": money(decision.get("totalCreditUsd")),
                "risk": money(decision.get("totalMaxRiskUsd")),
                "prob": (num(decision.get("averageProbNoTouch")) or 0) * 100 if decision.get("averageProbNoTouch") is not None else None,
                "touch": ml_touch,
                "close": num(context.get("close")),
                "inside": ml_inside,
                "nearest": ml_nearest,
                "close_pct": ml_close_pct,
                "profiles": profiles,
                "confidence": decision.get("confidence") or "-",
            }
        )

    return ml_dates, benchmark_rows, ml_rows


def weekly_summary(rows, kind):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["week"]].append(row)
    summary = []
    for week in sorted(buckets):
        rows_w = buckets[week]
        if kind == "benchmark":
            ops = [r for r in rows_w if r["status"] == "active"]
            pnl_values = [r["pnl"] for r in ops if r["pnl"] is not None]
            max_mult = [r["max_mult"] for r in ops if r["max_mult"] is not None]
            extra = {
                "pnl": sum(pnl_values) if pnl_values else None,
                "max_mult": max(max_mult) if max_mult else None,
            }
        else:
            ops = [r for r in rows_w if r["action"] != "no_operar"]
            risks = [r["risk"] for r in ops if r["risk"] is not None and r["risk"] > 0]
            extra = {
                "avg_risk": sum(risks) / len(risks) if risks else None,
                "max_mult": None,
            }
        wins = [r for r in ops if r["touch"] is False]
        touches = [r for r in ops if r["touch"] is True]
        credits = [r["credit"] for r in ops if r["credit"] is not None]
        summary.append(
            {
                "week": week,
                "sessions": len(rows_w),
                "ops": len(ops),
                "wins": len(wins),
                "touches": len(touches),
                "wr": len(wins) / len(ops) * 100 if ops else None,
                "credit": sum(credits) if credits else None,
                "avg_credit": sum(credits) / len(credits) if credits else None,
                **extra,
            }
        )
    return summary


def totals(rows, kind):
    if kind == "benchmark":
        ops = [r for r in rows if r["status"] == "active"]
        pnl = [r["pnl"] for r in ops if r["pnl"] is not None]
    else:
        ops = [r for r in rows if r["action"] != "no_operar"]
        pnl = []
    wins = [r for r in ops if r["touch"] is False]
    touches = [r for r in ops if r["touch"] is True]
    credits = [r["credit"] for r in ops if r["credit"] is not None]
    risks = [r.get("risk") for r in ops if r.get("risk") is not None and r.get("risk") > 0]
    return {
        "sessions": len(rows),
        "ops": len(ops),
        "no_trade": len(rows) - len(ops),
        "wins": len(wins),
        "touches": len(touches),
        "wr": len(wins) / len(ops) * 100 if ops else None,
        "credit": sum(credits) if credits else None,
        "avg_credit": sum(credits) / len(credits) if credits else None,
        "pnl": sum(pnl) if pnl else None,
        "avg_risk": sum(risks) / len(risks) if risks else None,
    }


def yes_no(value):
    if value is True:
        return "Si"
    if value is False:
        return "No"
    return "-"


def touch_text(row):
    if row.get("touch") is True:
        side = row.get("touch_side")
        return "Si " + side if side and side != "-" else "Si"
    if row.get("touch") is False:
        return "No"
    return "-"


def make_table(data, col_widths, header_bg=colors.HexColor("#111111"), font_size=6.4):
    table = Table(data, repeatRows=1, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), font_size),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d0d0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f3")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 10 * mm, "SPX-IC audit report - benchmark vs ML Shadow")
    canvas.drawRightString(279 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def paragraph(text, style):
    return Paragraph(text, style)


def generate_pdf():
    os.makedirs(OUT_DIR, exist_ok=True)
    ml_dates, benchmark_rows, ml_rows = build_rows()
    benchmark_total = totals(benchmark_rows, "benchmark")
    ml_total = totals(ml_rows, "ml")
    benchmark_week = weekly_summary(benchmark_rows, "benchmark")
    ml_week = weekly_summary(ml_rows, "ml")

    page_size = landscape(A4)
    doc = BaseDocTemplate(
        REPORT_FILE,
        pagesize=page_size,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=15 * mm,
        title="SPX-IC Auditoria Benchmark vs ML Shadow",
        author="Codex / E. M. C",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=page_footer)])

    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleAudit", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, alignment=TA_CENTER, textColor=colors.HexColor("#111111"), spaceAfter=10)
    subtitle = ParagraphStyle("SubtitleAudit", parent=styles["BodyText"], fontSize=9, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=14)
    section = ParagraphStyle("SectionAudit", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor("#111111"), spaceBefore=8, spaceAfter=8)
    body = ParagraphStyle("BodyAudit", parent=styles["BodyText"], fontSize=8.5, leading=12, textColor=colors.HexColor("#333333"), alignment=TA_LEFT, spaceAfter=8)

    story = []
    story.append(paragraph("SPX-IC Auditoria Comparativa", title))
    story.append(paragraph("Benchmark SpotGamma vs ML Shadow - periodo comun: " + f"{ml_dates[0]} a {ml_dates[-1]} - generado {datetime.now():%Y-%m-%d %H:%M}", subtitle))
    story.append(paragraph("Criterios usados", section))
    story.append(paragraph("Touch significa que el SPX toca durante la sesion el strike vendido. Cierre dentro banda significa que el cierre queda en zona segura respecto a las patas vendidas: entre put vendida y call vendida si hay Iron Condor, por debajo de la call vendida si solo hay call, o por encima de la put vendida si solo hay put. Dist. cierre % es positiva si el cierre queda dentro/seguro y negativa si cierra fuera; se mide frente a la banda vendida mas cercana.", body))

    summary_data = [
        ["Modelo", "Sesiones", "Operaciones", "No operar", "No-touch", "Touches", "WR", "Prima total", "Prima media", "P&L cierre", "Riesgo medio"],
        ["Benchmark", fmt(benchmark_total["sessions"]), fmt(benchmark_total["ops"]), fmt(benchmark_total["no_trade"]), fmt(benchmark_total["wins"]), fmt(benchmark_total["touches"]), fmt(benchmark_total["wr"], 1) + "%", fmt_money(benchmark_total["credit"]), fmt_money(benchmark_total["avg_credit"]), fmt_money(benchmark_total["pnl"]), "-"],
        ["ML Shadow", fmt(ml_total["sessions"]), fmt(ml_total["ops"]), fmt(ml_total["no_trade"]), fmt(ml_total["wins"]), fmt(ml_total["touches"]), fmt(ml_total["wr"], 1) + "%", fmt_money(ml_total["credit"]), fmt_money(ml_total["avg_credit"]), "-", fmt_money(ml_total["avg_risk"])],
    ]
    story.append(make_table(summary_data, [28 * mm, 18 * mm, 22 * mm, 20 * mm, 19 * mm, 18 * mm, 16 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm], font_size=7.0))
    story.append(Spacer(1, 8))
    story.append(paragraph("Lectura rapida: el benchmark mantiene un WR no-touch ligeramente superior, mientras que ML Shadow genera mas operaciones y mas prima estimada. La lectura todavia es provisional: N=24 sesiones.", body))

    story.append(PageBreak())
    story.append(paragraph("Tabla 1 - Benchmark SpotGamma", section))
    benchmark_table = [["Fecha", "Estado", "CW/PW", "Call", "Put", "Prima", "Touch", "Cierre", "Dentro", "Banda cercana", "Dist. cierre", "P&L", "Max x", "Skip"]]
    for r in benchmark_rows:
        benchmark_table.append(
            [
                r["date"],
                r["status"],
                f"{fmt(r['cw'])}/{fmt(r['pw'])}",
                r["call"],
                r["put"],
                fmt_money(r["credit"]),
                touch_text(r),
                fmt(r["close"]),
                yes_no(r["inside"]),
                r["nearest"] or "-",
                fmt_pct(r["close_pct"]),
                fmt_money(r["pnl"]),
                fmt(r["max_mult"], 2),
                r["skipped"],
            ]
        )
    story.append(make_table(benchmark_table, [19 * mm, 18 * mm, 21 * mm, 22 * mm, 22 * mm, 17 * mm, 17 * mm, 17 * mm, 16 * mm, 22 * mm, 22 * mm, 18 * mm, 17 * mm, 17 * mm], font_size=6.0))

    story.append(Spacer(1, 10))
    story.append(paragraph("Resumen semanal Benchmark", section))
    bw_table = [["Semana", "Ses.", "Ops", "No-touch", "Touches", "WR", "Prima total", "Prima media", "P&L", "Max x"]]
    for r in benchmark_week:
        bw_table.append([r["week"], fmt(r["sessions"]), fmt(r["ops"]), fmt(r["wins"]), fmt(r["touches"]), fmt(r["wr"], 1) + "%" if r["wr"] is not None else "-", fmt_money(r["credit"]), fmt_money(r["avg_credit"]), fmt_money(r["pnl"]), fmt(r["max_mult"], 2)])
    story.append(make_table(bw_table, [25 * mm, 16 * mm, 16 * mm, 20 * mm, 20 * mm, 18 * mm, 24 * mm, 24 * mm, 22 * mm, 18 * mm], font_size=7.0))

    story.append(PageBreak())
    story.append(paragraph("Tabla 2 - ML Shadow V1.1", section))
    story.append(paragraph("Nota: ML Shadow esta en paper mode. La prima y el riesgo son estimaciones iniciales de la estructura propuesta; todavia no hay P&L intradia monitorizado para sus propios strikes.", body))
    ml_table = [["Fecha", "Accion", "Call sell/buy", "Put sell/buy", "Prima", "Riesgo", "Prob.", "Touch", "Cierre", "Dentro", "Banda cercana", "Dist. cierre", "Perfil"]]
    for r in ml_rows:
        ml_table.append(
            [
                r["date"],
                r["action"],
                r["call"],
                r["put"],
                fmt_money(r["credit"]),
                fmt_money(r["risk"]),
                fmt(r["prob"], 1) + "%" if r["prob"] is not None else "-",
                touch_text(r),
                fmt(r["close"]),
                yes_no(r["inside"]),
                r["nearest"] or "-",
                fmt_pct(r["close_pct"]),
                r["profiles"],
            ]
        )
    story.append(make_table(ml_table, [19 * mm, 21 * mm, 28 * mm, 28 * mm, 18 * mm, 20 * mm, 17 * mm, 16 * mm, 17 * mm, 16 * mm, 22 * mm, 22 * mm, 28 * mm], font_size=5.9))

    story.append(Spacer(1, 10))
    story.append(paragraph("Resumen semanal ML Shadow", section))
    mw_table = [["Semana", "Ses.", "Ops", "No-touch", "Touches", "WR", "Prima total", "Prima media", "Riesgo medio"]]
    for r in ml_week:
        mw_table.append([r["week"], fmt(r["sessions"]), fmt(r["ops"]), fmt(r["wins"]), fmt(r["touches"]), fmt(r["wr"], 1) + "%" if r["wr"] is not None else "-", fmt_money(r["credit"]), fmt_money(r["avg_credit"]), fmt_money(r["avg_risk"])])
    story.append(make_table(mw_table, [25 * mm, 16 * mm, 16 * mm, 20 * mm, 20 * mm, 18 * mm, 24 * mm, 24 * mm, 24 * mm], font_size=7.0))

    doc.build(story)
    return REPORT_FILE


if __name__ == "__main__":
    print(generate_pdf())
