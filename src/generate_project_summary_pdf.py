"""Full project technical summary PDF for the E-Commerce Churn AI platform."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak, ListFlowable, ListItem


def _styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        'T', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16,
        leading=20, alignment=TA_CENTER, textColor=colors.HexColor('#1A365D'), spaceAfter=8
    )
    sub = ParagraphStyle(
        'S', parent=styles['Normal'], fontName='Helvetica', fontSize=10,
        leading=14, alignment=TA_CENTER, textColor=colors.HexColor('#4A5568'), spaceAfter=16
    )
    h1 = ParagraphStyle(
        'H1', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13,
        leading=17, textColor=colors.HexColor('#2B6CB0'), spaceBefore=14, spaceAfter=8
    )
    h2 = ParagraphStyle(
        'H2', parent=styles['Heading3'], fontName='Helvetica-Bold', fontSize=11,
        leading=15, textColor=colors.HexColor('#2C5282'), spaceBefore=10, spaceAfter=6
    )
    body = ParagraphStyle(
        'B', parent=styles['Normal'], fontName='Helvetica', fontSize=10,
        leading=14, alignment=TA_JUSTIFY, textColor=colors.HexColor('#2D3748'), spaceAfter=8
    )
    bullet = ParagraphStyle(
        'Bu', parent=body, leftIndent=14, bulletIndent=4, alignment=TA_LEFT
    )
    return title, sub, h1, h2, body, bullet


def _table(rows):
    t = Table(rows, colWidths=[160, 340])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F7FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#CBD5E0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))
    return t


def generate_project_summary_pdf(output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    title, sub, h1, h2, body, bullet = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=48
    )
    story = []
    story.append(Paragraph("E-Commerce Churn AI Decision Support Platform", title))
    story.append(Paragraph(
        "Project Technical Summary &amp; Methodology Report<br/>"
        "Muhammad Tehmas (25MEIT003) &mdash; Mehran University of Engineering and Technology (MUET)<br/>"
        "Supervisors: Prof. Dr. Shahnawaz Talpur | Engr. Madeha Memon",
        sub
    ))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#2B6CB0')))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Project Overview", h1))
    story.append(Paragraph(
        "This project is a web-based decision-support system that predicts whether an e-commerce customer "
        "is likely to leave (churn). Shopkeepers usually discover churn only after the customer has already gone. "
        "The platform trains an interpretable glassbox model on a public Kaggle e-commerce churn dataset, then "
        "applies that trained model to a newly uploaded store dataset (for example EasyBazar). The uploaded file "
        "is not expected to contain a Churn label. The application writes two prediction columns onto a copy of "
        "the original file: binary Churn (0/1) and Churn_Prediction_Percentage.",
        body
    ))

    story.append(Paragraph("2. What Was Built (Web Application)", h1))
    story.append(Paragraph(
        "The user-facing product is a Streamlit web app (typically http://localhost:8501). It is a Python dashboard, "
        "not a separate React/Next.js frontend. Plotly is used for interactive charts. Pages include:",
        body
    ))
    for item in [
        "<b>Executive AI Dashboard:</b> predicted KPIs (total customers, predicted churners, non-churners, churn rate, average probability, high-risk customers) plus product, city-tier, COD, and RFM charts.",
        "<b>Retargeting Action Center:</b> customer risk table, export CSV, and one-click outreach drafts.",
        "<b>AI Model Performance:</b> Kaggle held-out accuracy/precision/recall/F1/ROC, confusion matrix; uploaded view shows applied predictions plus the same EBM exam scores.",
        "<b>Project Methodology:</b> thesis framing and feature dictionary.",
        "<b>Complete Dataset Explorer:</b> inspect uploaded vs Kaggle tables and download the enriched file.",
        "<b>Data Hub &amp; AI Retraining:</b> merge/replace master data and regenerate predictions.",
    ]:
        story.append(Paragraph("&bull; " + item, bullet))

    story.append(Paragraph("3. Technologies Used", h1))
    story.append(_table([
        ["Layer", "Technology"],
        ["Web UI", "Streamlit, Plotly, custom CSS (SaaS layout)"],
        ["Language", "Python 3"],
        ["Data", "pandas, numpy, openpyxl"],
        ["ML (production)", "Microsoft InterpretML Explainable Boosting Machine (glassbox)"],
        ["Evaluation", "scikit-learn (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)"],
        ["Split", "80/20 stratified train-test on Kaggle Churn"],
        ["Imputation (train)", "Median fallback (PyTorch autoencoder if torch is installed)"],
        ["Model file", "final_model.sav (pickle)"],
        ["Inference pipeline", "inference_pipeline.sav (feature list, medians, 0.5 threshold)"],
        ["Reporting", "ReportLab PDF"],
    ]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Production predictions do not use neural nets or random forests. Those appear only in an optional "
        "comparative-analysis script as a glassbox-vs-blackbox thesis proof. Live scoring is EBM only.",
        body
    ))

    story.append(Paragraph("4. Training Dataset (Kaggle)", h1))
    story.append(Paragraph(
        "Learning uses the Kaggle E-Commerce Customer Churn file in this repo: "
        "<b>E Commerce Dataset Updated.xlsx</b> (sheet E Comm). Raw shape is 5,630 rows and 20 columns, "
        "including a ground-truth <b>Churn</b> column (1 = left, 0 = stayed). Overall churn rate is about 16.8%.",
        body
    ))
    story.append(Paragraph(
        "The full Kaggle file also contains columns such as CityTier, Gender, CashbackAmount, PreferedOrderCat, "
        "and MaritalStatus. Those fields are <b>not</b> used for training in the current production model. "
        "They may still appear on the dashboard when present in an uploaded file, because charts group predictions "
        "after scoring. They do not teach the EBM.",
        body
    ))

    story.append(Paragraph("5. Learning Design: Inputs vs Output", h1))
    story.append(Paragraph("5.1 Output (target / label)", h2))
    story.append(Paragraph(
        "The only training target is Kaggle <b>Churn</b>. It is never used as an input feature (that would leak "
        "the answer). The model learns P(Churn = 1 | inputs).",
        body
    ))
    story.append(Paragraph("5.2 Inputs (features the EBM actually learns from)", h2))
    story.append(_table([
        ["Column", "Meaning"],
        ["Tenure", "How long the customer has been with the store (months)"],
        ["OrderCount", "How many orders they placed (frequency)"],
        ["DaySinceLastOrder", "Days since last purchase (recency)"],
        ["Complain", "1 if they filed a complaint, else 0"],
    ]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Train set: 4,504 rows. Held-out test set: 1,126 rows. Stratified so both splits keep ~16.8% churners.",
        body
    ))

    story.append(Paragraph("6. How the Glassbox EBM Learns", h1))
    story.append(Paragraph(
        "Explainable Boosting Machine builds a scorecard for each input (and some pairwise interactions), then adds "
        "them: total score = intercept + f(Tenure) + f(OrderCount) + f(DaySinceLastOrder) + f(Complain) + interactions. "
        "If the probability from that score is at least 50%, predicted Churn = 1, else 0. Unlike a neural net, "
        "each feature contribution can be inspected (glassbox).",
        body
    ))
    story.append(Paragraph(
        "Observed Kaggle patterns during training include: complainers churn at about 31.7% vs 11.0% for non-complainers; "
        "churners have much shorter tenure (about 3.8 months vs 11.4). The model uses those relationships, not hardcoded if-else rules.",
        body
    ))

    story.append(Paragraph("7. Evaluation on Kaggle (Is It Correct?)", h1))
    story.append(Paragraph(
        "Yes. Metrics are computed on the held-out test set that the model did not train on, using true Kaggle Churn "
        "labels versus EBM predictions. Current scores:",
        body
    ))
    story.append(_table([
        ["Metric", "Value and meaning"],
        ["Accuracy", "88.45% overall correct (safe + churn)"],
        ["Precision", "69.23% of predicted churners actually churned"],
        ["Recall", "56.84% of actual churners were caught"],
        ["F1", "62.43% balance of precision and recall"],
        ["ROC-AUC", "90.95% ranking quality (well above 50% coin-toss)"],
        ["Confusion", "TN 888, FP 48, FN 82, TP 108 (1,126 test rows)"],
    ]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("7.1 Why precision and recall look lower than accuracy", h2))
    story.append(Paragraph(
        "This is expected and the evaluation is not broken. About 83% of customers stay. A model that is strong on "
        "the majority class gets high accuracy from many true negatives (888). Precision and recall focus on the "
        "minority class (only ~190 actual churners in the test set). Missing 82 churners (false negatives) pulls "
        "recall to 57%. 48 false alarms pull precision to 69%. ROC-AUC of 91% shows the model still ranks risky "
        "customers well. Using only four features (no cashback, category, or marital status) also limits recall "
        "compared with an earlier 12-feature version that scored ~93% accuracy.",
        body
    ))

    story.append(Paragraph("8. What Happens When a New Dataset Is Uploaded", h1))
    story.append(Paragraph(
        "The uploaded file (CSV/Excel) is not required to contain Churn. The app does not treat missing Churn as an error. "
        "Original columns are preserved. A separate feature matrix is built by matching Tenure, OrderCount, "
        "DaySinceLastOrder, and Complain (case-insensitive). Missing model inputs are filled with training medians "
        "only inside that matrix, not by rewriting the user's cells. Then:",
        body
    ))
    story.append(Paragraph(
        "churn_probability = model.predict_proba(X)[:, 1]<br/>"
        "Churn_Prediction_Percentage = round(probability * 100, 1)<br/>"
        "Churn = 1 if probability &gt;= 0.5 else 0",
        body
    ))
    story.append(Paragraph(
        "Those two columns are appended. The dashboard then shows predicted analytics (counts, rate, average probability, "
        "high-risk share where probability is at least 70%, risk histogram, product/city charts). Download exports "
        "original columns plus Churn plus Churn_Prediction_Percentage. True accuracy cannot be computed on the upload "
        "because there are no actual outcomes; the Kaggle exam scores describe the model that was applied.",
        body
    ))

    story.append(Paragraph("9. End-to-End Pipeline", h1))
    for i, step in enumerate([
        "Load Kaggle Excel and identify Churn as the training target.",
        "Clean/impute training fields; keep only the four learning columns for X.",
        "Stratified 80/20 split.",
        "Train glassbox EBM; evaluate on the 1,126-row test set.",
        "Save final_model.sav and inference_pipeline.sav.",
        "User uploads a store dataset in the Streamlit sidebar.",
        "Map the four inputs; apply the same threshold and probability output.",
        "Write Churn and Churn_Prediction_Percentage; display dashboards; allow CSV download.",
    ], start=1):
        story.append(Paragraph(f"{i}. {step}", bullet))

    story.append(Paragraph("10. Key Source Files", h1))
    story.append(_table([
        ["File", "Role"],
        ["app.py", "Streamlit web application"],
        ["src/preprocess.py", "Kaggle load, clean, 4-feature split"],
        ["retrain.py", "Train EBM and save pipeline"],
        ["src/inference.py", "Upload scoring without mutating originals"],
        ["final_model.sav", "Trained glassbox classifier"],
        ["inference_pipeline.sav", "Feature names, medians, 0.5 cutoff"],
        ["E Commerce Dataset Updated.xlsx", "Kaggle training source"],
    ]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This document describes the production workflow as implemented: Kaggle-supervised glassbox learning on four "
        "behavioral columns, then inference that creates Churn and Churn_Prediction_Percentage for unlabeled uploads.",
        body
    ))

    doc.build(story)
    print(f"[+] Wrote {output_path}")


if __name__ == "__main__":
    out = os.path.join("reports", "Project_Technical_Summary.pdf")
    generate_project_summary_pdf(out)
    generate_project_summary_pdf("Project_Technical_Summary.pdf")
