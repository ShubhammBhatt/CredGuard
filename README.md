# CreditGuard AI – Transaction Fraud Detection & Risk Console

An end-to-end machine learning fraud scoring application built with **scikit-learn / imbalanced-learn**, **FastAPI**, **SQLite**, and **Streamlit**. The project separates model training, REST API risk evaluation, audit logging, and interactive console visualization into clean, modular components.

## Key Features
- **Modular ML Pipeline:** Generates synthetic transaction datasets, preprocesses numerical and categorical features (`StandardScaler` + `OneHotEncoder`), handles class imbalance with `SMOTE` inside an `ImbPipeline`, trains and compares candidate models, and saves `artifacts/model.joblib` and `artifacts/metadata.json`.
- **FastAPI Backend (`src/api.py`):** Validates incoming transaction payloads with Pydantic, computes fraud probabilities, maps scores to actionable risk decisions (**Approve**, **Flag for Review**, **Block**), and logs predictions to SQLite.
- **SQLite Audit Log (`src/database.py`):** Persists scored transactions, probabilities, and decisions in `data/creditguard.db` for historical review and KPI tracking.
- **Streamlit Risk Console (`app/streamlit_app.py`):** Displays real-time KPI counts, one-click preset scenarios (*Legit Grocery Swipe*, *Borderline Review Case*, *High-Risk Foreign Electronics*), live risk scoring, probability bars, and recent logged predictions.

## Project Structure
```text
.
├── .streamlit/
│   └── config.toml         # Theme configuration (colors, borders, radii)
├── app/
│   └── streamlit_app.py    # Interactive risk console UI
├── artifacts/
│   ├── model.joblib        # Saved trained ML pipeline
│   └── metadata.json       # Model metadata and training info
├── data/
│   ├── transactions.csv    # Generated transaction dataset (local runtime)
│   └── creditguard.db      # SQLite prediction audit log (local runtime)
├── src/
│   ├── generate_data.py    # Synthetic dataset generator
│   ├── train.py            # Preprocessing, SMOTE, model selection & export
│   ├── database.py         # SQLite initialization and logging helpers
│   └── api.py              # FastAPI endpoints (/health, /predict, /predictions/recent)
├── .gitignore
├── requirements.txt
└── README.md