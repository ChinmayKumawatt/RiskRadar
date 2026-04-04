# RiskRadar — Multi-Disease Health Prediction System

RiskRadar is an end-to-end Machine Learning system designed to predict the risk of multiple chronic diseases using user health data. It combines a modular ML pipeline, a user-friendly interface, and real-time predictions to provide actionable health insights.

---

## Problem Statement

Chronic diseases such as heart disease, diabetes, hypertension, thyroid disorders, and chronic kidney disease are often diagnosed late due to lack of early screening and awareness.

There is a need for a unified system that:

* Predicts disease risks at an early stage
* Supports multiple diseases within a single platform
* Provides interpretable results to users
* Generates a consolidated health report

---

## Solution

RiskRadar addresses this problem by:

* Predicting multiple diseases using trained ML models
* Providing probability-based risk scores
* Generating a downloadable health report (PDF)
* Offering a structured and intuitive user input flow
* Using a modular, scalable ML pipeline architecture

---

## Live Application

The application is deployed and accessible online:

**Live URL:** https://riskradar-90vo.onrender.com/

---

## Key Features

* Multi-disease prediction:

  * Heart Disease
  * Type 2 Diabetes
  * Hypertension
  * Thyroid Disorders
  * Chronic Kidney Disease

* Step-by-step user input flow:

  * General Information
  * Lifestyle Factors
  * Disease-specific inputs

* Real-time predictions with probability scores

* Threshold-based classification

* Downloadable health report (PDF) including:

  * User input summary
  * Disease-wise risk analysis
  * Final predictions

* FastAPI-based backend

* Config-driven ML pipeline

---

## System Architecture

The system follows a production-style ML pipeline:

Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation → Deployment → Prediction Pipeline

### Multi-Disease Design

Each disease:

* Uses a separate dataset
* Has its own feature set
* Stores independent model artifacts

---

## Tech Stack

**Backend**

* Python
* FastAPI

**Machine Learning**

* Scikit-learn
* Pandas
* NumPy

**Frontend**

* HTML / CSS / JavaScript

**Deployment**

* Render

---

## Project Structure

```
RiskRadar/
│
├── artifacts/                  # Saved models and preprocessors
│   ├── heart/
│   ├── diabetes/
│   └── ...
│
├── src/
│   ├── components/             # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   ├── exception.py
│   │   └── common.py
│
├── app.py / main.py            # FastAPI entry point
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Prediction Workflow

User Input → Data Preprocessing → Model Selection → Probability Prediction → Threshold Evaluation → Result Output → PDF Report Generation

---

## Output

The system provides:

* Disease-wise predictions
* Risk probabilities
* Final classification (High / Low Risk)
* Downloadable PDF health report

---

## ML Pipeline Highlights

* Config-driven architecture
* Feature selection per disease
* Data validation (schema and consistency checks)
* Preprocessing pipelines (encoding and scaling)
* Model training and evaluation
* Threshold tuning

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ChinmayKumawatt/RiskRadar.git
cd RiskRadar
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
uvicorn app:app --reload
```

### 5. Open in browser

```
http://127.0.0.1:8000
```

---

## Future Improvements

* Explainability using SHAP
* User authentication and profile management
* Cloud database integration
* Mobile responsiveness improvements
* Model monitoring and retraining

---

## Author

Chinmay Kumawat
GitHub: https://github.com/ChinmayKumawatt

---

## License

This project is open-source and available under the MIT License.
