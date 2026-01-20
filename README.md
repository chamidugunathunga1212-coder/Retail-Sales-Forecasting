# 🛒 Walmart Sales Forecasting – End-to-End ML & MLOps Project

A **production-ready machine learning system** for forecasting Walmart weekly sales, built using **industry-standard MLOps practices**.

This project demonstrates how machine learning systems are **designed, validated, trained, tuned, tracked, and deployed** in real-world environments — not just notebook experimentation.

---

## 🚀 Project Highlights

- End-to-end ML pipeline (Ingestion → Validation → Transformation → Training → Serving)
- YAML-driven configuration (schema, features, models)
- Data validation with schema & quality rules
- Time-aware feature engineering
- Hyperparameter tuning using `RandomizedSearchCV`
- Experiment tracking with MLflow
- Batch inference using FastAPI (CSV upload)
- Versioned artifacts + stable `latest/` serving path

---

## 🧠 Business Problem

Retailers like Walmart require accurate **weekly sales forecasts** to:
- Optimize inventory planning
- Reduce stockouts and overstock
- Improve supply-chain efficiency

This system predicts **Weekly_Sales** for each **Store–Department–Date** combination using:
- Historical sales data
- Holiday indicators
- Weather & fuel prices
- Economic indicators
- Store metadata

---

## 🏗️ System Architecture

            ┌────────────┐
            │ Raw CSV    │
            │ (Kaggle)   │
            └─────┬──────┘
                  ↓
          ┌─────────────────┐
          │ Data Ingestion  │
          └─────┬───────────┘
                ↓
       ┌──────────────────────┐
       │ Data Validation       │
       │ (schema + quality)    │
       └─────┬────────────────┘
             ↓
    ┌─────────────────────────┐
    │ Feature Engineering       │
    │ (YAML-driven)             │
    └─────┬───────────────────┘
          ↓
 ┌────────────────────────────┐
 │ Data Transformation         │
 │ (ColumnTransformer)         │
 └─────┬──────────────────────┘
       ↓



---

## 🧪 Data Validation (Industry-Style)

- Required & optional column validation
- Missing value threshold checks
- Duplicate handling policy
- Target sanity checks (negative sales handling)
- Date parsing validation

📄 Output: artifacts/data_validation/validation_report.json



---

## 🧠 Feature Engineering

- Fully **YAML-driven**
- Numerical & categorical pipelines
- One-hot encoding
- Median / most-frequent imputation
- Time-based features:
  - year, month, weekofyear
  - dayofweek
  - month start / end flags
- Log transformation on target
- Negative target handling (drop policy)

✔ Ensures **training–serving parity**

---

## 🤖 Model Training & Hyperparameter Tuning

**Models**
- Random Forest
- Gradient Boosting
- (Optional) XGBoost

**Tuning**
- `RandomizedSearchCV`
- Cross-validation
- Configurable via `model.yaml`

**Metrics**
- RMSE (log scale)
- RMSE (original sales scale)

---

## 📈 Experiment Tracking

MLflow is used for:
- Parameter logging
- Metric comparison
- Best model selection

```bash
mlflow ui

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn
- MLflow
- FastAPI
- YAML-based configs


## 🌐 Inference API

- `GET /health`
- `POST /predict_csv` → Upload CSV, download predictions CSV

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python main.py
uvicorn app:app --reload

http://127.0.0.1:8000/docs

