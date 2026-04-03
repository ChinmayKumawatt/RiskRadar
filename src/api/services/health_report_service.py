from io import BytesIO
from textwrap import wrap

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


class HealthReportService:
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service

    def generate_pdf(self, patient_name, request_payload):
        predictions = self.prediction_service.predict_all(request_payload)
        assessments = self._build_general_assessments(request_payload)

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 50

        y = self._draw_heading(
            pdf,
            y,
            "RiskRadar General Health Report",
            f"Patient: {patient_name or 'Not provided'}",
        )

        y = self._draw_section_title(pdf, y, "General Health Review")
        for assessment in assessments:
            y = self._draw_bullet_block(
                pdf,
                y,
                f"{assessment['label']}: {assessment['status']}",
                assessment["details"],
            )

        y = self._draw_section_title(pdf, y, "Prediction Summary")
        for disease_name, result in predictions["predictions"].items():
            title = disease_name.upper() if disease_name == "ckd" else disease_name.title()
            summary = (
                "Higher risk pattern detected."
                if result["risk_detected"]
                else "Lower risk pattern detected."
            )
            detail = (
                f"Prediction label: {result['prediction_label']}. "
                f"Risk probability: {self._format_percent(result.get('risk_probability'))}."
            )
            y = self._draw_bullet_block(pdf, y, title, f"{summary} {detail}")

        if predictions["skipped"]:
            y = self._draw_section_title(pdf, y, "Skipped Predictions")
            for disease_name, reason in predictions["skipped"].items():
                y = self._draw_bullet_block(pdf, y, disease_name.title(), reason)

        y = self._draw_section_title(pdf, y, "Reference Standards")
        references = [
            "BMI: CDC Adult BMI Categories",
            "Blood Pressure and Resting Heart Rate: American Heart Association",
            "Fasting Blood Sugar: CDC Diabetes Testing guidance",
            "Total Cholesterol: MedlinePlus cholesterol levels guidance",
        ]
        for reference in references:
            y = self._draw_bullet_block(pdf, y, reference, "")

        y = self._draw_section_title(pdf, y, "Important Note")
        y = self._draw_paragraph(
            pdf,
            y,
            "This report is an informational screening summary only. It is not a diagnosis "
            "and should not replace medical advice from a qualified healthcare professional.",
        )

        pdf.save()
        buffer.seek(0)
        return buffer

    def _build_general_assessments(self, payload):
        assessments = []

        age = payload.get("age")
        if age is not None:
            assessments.append({
                "label": "Age",
                "status": f"{age} years",
                "details": "Used as a general context factor in the screening.",
            })

        bmi = payload.get("bmi")
        if bmi is not None:
            if bmi < 18.5:
                status = "Below healthy range"
            elif bmi < 25:
                status = "Within healthy range"
            elif bmi < 30:
                status = "Above healthy range"
            else:
                status = "Obesity range"
            assessments.append({
                "label": "BMI",
                "status": status,
                "details": f"Recorded BMI: {bmi}. CDC adult healthy range is 18.5 to less than 25.",
            })

        blood_pressure = payload.get("bloodpressure") or payload.get("trestbps")
        if blood_pressure is not None:
            if blood_pressure < 90:
                status = "Low reading"
            elif blood_pressure < 120:
                status = "Normal"
            elif blood_pressure < 130:
                status = "Elevated"
            elif blood_pressure < 140:
                status = "Stage 1 high range"
            elif blood_pressure <= 180:
                status = "Stage 2 high range"
            else:
                status = "Severely high reading"
            assessments.append({
                "label": "Blood Pressure",
                "status": status,
                "details": (
                    f"Recorded value: {blood_pressure}. "
                    "AHA normal systolic reading is less than 120."
                ),
            })

        cholesterol = payload.get("totchol") or payload.get("chol")
        if cholesterol is not None:
            if cholesterol < 200:
                status = "Desirable"
            elif cholesterol < 240:
                status = "Borderline high"
            else:
                status = "High"
            assessments.append({
                "label": "Total Cholesterol",
                "status": status,
                "details": (
                    f"Recorded value: {cholesterol} mg/dL. "
                    "MedlinePlus lists less than 200 mg/dL as the healthy total cholesterol level."
                ),
            })

        glucose = payload.get("glucose")
        if glucose is not None:
            if glucose <= 99:
                status = "Normal fasting range"
            elif glucose <= 125:
                status = "Prediabetes range"
            else:
                status = "Diabetes-range high"
            assessments.append({
                "label": "Blood Sugar",
                "status": status,
                "details": (
                    f"Recorded value: {glucose} mg/dL. "
                    "CDC fasting blood sugar guidance treats 99 or below as normal."
                ),
            })

        heart_rate = payload.get("heartrate")
        if heart_rate is not None:
            if heart_rate < 60:
                status = "Below usual adult resting range"
            elif heart_rate <= 100:
                status = "Within usual adult resting range"
            else:
                status = "Above usual adult resting range"
            assessments.append({
                "label": "Resting Heart Rate",
                "status": status,
                "details": (
                    f"Recorded value: {heart_rate} bpm. "
                    "AHA notes that 60 to 100 bpm is normal for most adults at rest."
                ),
            })

        smoking = payload.get("cigsperday")
        if smoking is not None:
            if smoking == 0:
                status = "Non-smoker"
            elif smoking <= 5:
                status = "Light smoking pattern"
            elif smoking <= 15:
                status = "Moderate smoking pattern"
            else:
                status = "Heavy smoking pattern"
            assessments.append({
                "label": "Smoking",
                "status": status,
                "details": f"Recorded value: {smoking} cigarettes per day.",
            })

        return assessments

    @staticmethod
    def _format_percent(probability):
        if probability is None:
            return "Not available"
        return f"{probability * 100:.1f}%"

    def _draw_heading(self, pdf, y, title, subtitle):
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(50, y, title)
        y -= 22
        pdf.setFont("Helvetica", 11)
        pdf.drawString(50, y, subtitle)
        return y - 24

    def _draw_section_title(self, pdf, y, title):
        y = self._ensure_space(pdf, y, 40)
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y, title)
        return y - 18

    def _draw_bullet_block(self, pdf, y, title, detail):
        y = self._ensure_space(pdf, y, 56)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(60, y, f"- {title}")
        y -= 14
        if detail:
            y = self._draw_paragraph(pdf, y, detail, indent=76)
        return y - 8

    def _draw_paragraph(self, pdf, y, text, indent=50):
        pdf.setFont("Helvetica", 10)
        for line in wrap(text, width=92):
            y = self._ensure_space(pdf, y, 20)
            pdf.drawString(indent, y, line)
            y -= 13
        return y

    def _ensure_space(self, pdf, y, required_space):
        if y >= 50 + required_space:
            return y
        pdf.showPage()
        return A4[1] - 50
