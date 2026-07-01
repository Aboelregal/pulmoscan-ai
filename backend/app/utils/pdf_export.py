"""PDF export for finalized radiology reports, using reportlab."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime


def export_report_pdf(report, study, out_path):
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    styles = getSampleStyleSheet()
    header_style = ParagraphStyle("Header", parent=styles["Heading1"], textColor=colors.HexColor("#0f172a"))
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], textColor=colors.HexColor("#1d4ed8"))

    elements = [
        Paragraph("PulmoScan AI — Chest CT Report", header_style),
        Paragraph(f"Study UID: {study.study_uid}", styles["Normal"]),
        Paragraph(f"Generated: {datetime.utcnow().isoformat()}Z", styles["Normal"]),
        Spacer(1, 0.3 * inch),

        Paragraph("Findings", section_style),
        Paragraph(report.findings_text.replace("\n", "<br/>"), styles["Normal"]),
        Spacer(1, 0.2 * inch),

        Paragraph("Impression", section_style),
        Paragraph(report.impression_text.replace("\n", "<br/>"), styles["Normal"]),
        Spacer(1, 0.2 * inch),

        Paragraph("Recommendations", section_style),
        Paragraph(report.recommendations_text.replace("\n", "<br/>"), styles["Normal"]),
        Spacer(1, 0.2 * inch),

        Paragraph("Patient-Friendly Summary", section_style),
        Paragraph(report.patient_summary_text or "", styles["Normal"]),
        Spacer(1, 0.3 * inch),

        Paragraph(
            "This report was generated with AI assistance and must be reviewed "
            "and signed off by a licensed radiologist before clinical use.",
            styles["Italic"],
        ),
    ]

    if report.references_json:
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("References", section_style))
        data = [["Title", "Source"]] + [[r["title"], r["source"]] for r in report.references_json]
        table = Table(data, colWidths=[4 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        elements.append(table)

    doc.build(elements)
    return out_path
