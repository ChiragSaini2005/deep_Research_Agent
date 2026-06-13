# backend/utils/pdf_export.py

import re
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
)
from datetime import datetime


def _markdown_inline_to_reportlab(text: str) -> str:
    """Convert basic markdown inline syntax to reportlab XML markup."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)   # bold
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)       # italic
    return text


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontSize=24, spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"],
        fontSize=11, textColor="#666666", alignment=TA_CENTER, spaceAfter=24
    ))
    return styles


def markdown_to_pdf(title: str, query: str, body_markdown: str) -> bytes:
    """
    Converts a research report (title + markdown body) into a styled PDF.
    Returns raw PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=1 * inch, bottomMargin=1 * inch,
        leftMargin=1 * inch, rightMargin=1 * inch
    )
    styles = _build_styles()
    story = []

    # ── Title Page ──
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph(title, styles["ReportTitle"]))
    story.append(Paragraph(f"Research query: {query}", styles["ReportSubtitle"]))
    story.append(Paragraph(
        datetime.now().strftime("%B %d, %Y"), styles["ReportSubtitle"]
    ))
    story.append(PageBreak())

    # ── Body ──
    list_items = []

    def flush_list():
        if list_items:
            story.append(ListFlowable(
                [ListItem(Paragraph(item, styles["Normal"])) for item in list_items],
                bulletType="bullet", leftIndent=20
            ))
            story.append(Spacer(1, 8))
            list_items.clear()

    for line in body_markdown.splitlines():
        line = line.rstrip()

        if not line.strip():
            flush_list()
            story.append(Spacer(1, 6))
            continue

        if line.startswith("### "):
            flush_list()
            story.append(Paragraph(_markdown_inline_to_reportlab(line[4:]), styles["Heading3"]))
        elif line.startswith("## "):
            flush_list()
            story.append(Paragraph(_markdown_inline_to_reportlab(line[3:]), styles["Heading2"]))
        elif line.startswith("# "):
            flush_list()
            story.append(Paragraph(_markdown_inline_to_reportlab(line[2:]), styles["Heading1"]))
        elif line.startswith(("- ", "* ")):
            list_items.append(_markdown_inline_to_reportlab(line[2:]))
        else:
            flush_list()
            story.append(Paragraph(_markdown_inline_to_reportlab(line), styles["Normal"]))
            story.append(Spacer(1, 4))

    flush_list()

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes