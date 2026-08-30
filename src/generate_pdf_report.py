"""
PDF Report Generator for Phase 1: Data Preprocessing & Exploratory Analysis
Master Thesis: A User-Centric Business Intelligence Framework for Predictive Customer Churn in E-Commerce
"""

import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def generate_pdf(summary_file: str, output_pdf_path: str):
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary JSON not found at: {summary_file}")
        
    with open(summary_file, 'r') as f:
        data = json.load(f)
        
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=15,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=6
    )

    story = []
    
    # Header Section
    story.append(Paragraph("MASTER THESIS: PROJECT MASTER DOCUMENT", subtitle_style))
    story.append(Paragraph(data["project_title"], title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2B6CB0'), spaceBefore=0, spaceAfter=15))
    
    meta_text = (
        f"<b>Researcher:</b> {data['student']}<br/>"
        f"<b>Institution:</b> {data['university']}<br/>"
        f"<b>Supervisors:</b> {data['supervisors']}<br/>"
        f"<b>Current Phase:</b> Phase 1 Completed (Data Preprocessing & EDA)"
    )
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 10))
    
    # Section 1: Executive Summary & Objective
    story.append(Paragraph("1. Phase 1 Executive Summary & Dataset Profile", h1_style))
    summary_text = (
        "In this foundational phase, the baseline academic dataset (Kaggle E-Commerce Customer Churn) was ingested, "
        "cleaned, and prepared for interpretable machine learning modeling. A thorough exploratory data analysis (EDA) "
        "was conducted to identify data quality issues, missing distributions, and class imbalances."
    )
    story.append(Paragraph(summary_text, body_style))
    
    # Table of Dataset Metrics
    table_data = [
        ["Metric / Parameter", "Value / Description"],
        ["Dataset Source", data["dataset_name"]],
        ["Total Customer Records", f"{data['initial_records']:,} customers"],
        ["Predictive Features", f"{data['total_features']} behavioral & categorical features"],
        ["Target Variable", data["target_variable"]],
        ["Overall Churn Rate", f"{data['overall_churn_rate_pct']}% (Imbalanced Class Distribution)"]
    ]
    t = Table(table_data, colWidths=[2.2*inch, 4.3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (1,0), 10),
        ('BOTTOMPADDING', (0,0), (1,0), 6),
        ('TOPPADDING', (0,0), (1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#2D3748')),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # Section 2: Data Science Decision Rationale (Educational & Methodological)
    story.append(Paragraph("2. Methodological Decision Rationale (Why We Did What We Did)", h1_style))
    
    story.append(Paragraph("<b>Decision 1: Why Median Imputation for Missing Values?</b>", body_style))
    p1 = (
        "During inspection, 7 continuous variables showed missing values (e.g., DaySinceLastOrder had 307 missing, "
        "Tenure had 264 missing). In e-commerce customer behavior data, metrics are heavily right-skewed due to "
        "outliers (power buyers or extreme distances). If we used Mean (average) imputation, extreme outliers would "
        "pull the average upwards, distorting reality. <b>Median imputation (50th percentile)</b> was selected "
        "because it is robust against outliers and perfectly represents the typical customer's behavior."
    )
    story.append(Paragraph(p1, bullet_style))
    
    story.append(Paragraph("<b>Decision 2: Why Did We NOT Use One-Hot Encoding?</b>", body_style))
    p2 = (
        "Traditional machine learning pipelines convert categorical strings (like 'PreferredLoginDevice = Mobile') "
        "into multiple binary columns (e.g., 'Device_Mobile_1', 'Device_Laptop_0'). However, our primary thesis objective "
        "is <b>User-Centric Business Intelligence</b> for non-technical SME managers. We chose to preserve categorical strings "
        "as raw text because our Phase 2 model—<b>Explainable Boosting Machines (EBM)</b>—handles text natively and produces "
        "100% human-readable rules. One-Hot Encoding would destroy interpretability."
    )
    story.append(Paragraph(p2, bullet_style))
    
    story.append(Paragraph("<b>Decision 3: Why 80/20 Stratified Train-Test Splitting?</b>", body_style))
    p3 = (
        "Our dataset exhibits a natural class imbalance (~16.84% Churn vs 83.16% Retained). A simple random train-test split "
        "could accidentally assign an uneven percentage of churners to the training set. By using <b>Stratification "
        "(stratify=y)</b>, we guarantee that exact ~16.84% churn ratio is mirrored across both Train and Test splits, "
        "preventing sampling bias and ensuring fair model evaluation."
    )
    story.append(Paragraph(p3, bullet_style))
    story.append(Spacer(1, 10))
    
    # Section 3: Preprocessing & Imputation Results
    story.append(Paragraph("3. Preprocessing & Imputation Log", h1_style))
    
    imputation_rows = [["Feature Name", "Missing Count", "Imputed Median Value"]]
    for col, details in data["imputation_log"].items():
        imputation_rows.append([col, str(details["missing_filled"]), str(details["imputed_value"])])
        
    t2 = Table(imputation_rows, colWidths=[2.5*inch, 1.8*inch, 2.2*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#2C5282')),
        ('BACKGROUND', (2,0), (2,0), colors.HexColor('#2C5282')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#2D3748')),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
    ]))
    story.append(t2)
    story.append(Spacer(1, 15))
    
    # Section 4: Next Steps & Remaining Roadmap
    story.append(Paragraph("4. Remaining Thesis Roadmap (Phases 2 to 5)", h1_style))
    
    roadmap_items = [
        "<b>Phase 2 (Core Modeling & Comparative Analysis):</b> Train Microsoft's Explainable Boosting Machine (EBM) on X_train. Compare accuracy and ROC-AUC against a baseline Random Forest model to validate performance.",
        "<b>Phase 3 (Explainability Export & BI Dashboard):</b> Export global feature contributions and customer risk probabilities to structured CSVs. Design custom minimal backgrounds in Figma and build the interactive Power BI dashboard.",
        "<b>Phase 4 (Real-World Local Validation):</b> Ingest raw logistics and sales records from Urban Daraz. Perform RFM (Recency, Frequency, Monetary) feature engineering and stress-test the framework.",
        "<b>Phase 5 (Usability Evaluation & Defense):</b> Conduct System Usability Scale (SUS) and Technology Acceptance Model (TAM) surveys with SME managers. Finalize MUET thesis chapters for submission."
    ]
    for item in roadmap_items:
        story.append(Paragraph(f"• {item}", bullet_style))
        
    doc.build(story)
    print(f"[+] Successfully generated formal PDF Report at: {output_pdf_path}")

if __name__ == "__main__":
    summary_path = os.path.join("outputs", "phase1_summary.json")
    pdf_path = os.path.join("reports", "Phase_1_Project_Report.pdf")
    generate_pdf(summary_path, pdf_path)
