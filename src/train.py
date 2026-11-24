import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def main() -> None:
    # 1. Load the 10,000-row dataset 
    df = pd.read_csv("data/transactions.csv")

    target_col = "is_fraud"
    drop_cols = ["transaction_id", target_col]

    # X = 8 predictive features, y = target label (0 = legit, 1 = fraud)
    X = df.drop(columns=drop_cols)
    y = df[target_col].astype(int)

    # 2. Split BEFORE preprocessing/SMOTE so test data stays untouched (no data leakage)
    # stratify=y keeps the same ~1.51% fraud ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Handle numeric and text/categorical columns differently
    categorical_cols = ["merchant_category"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # 4. Candidate models to train and compare
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    best_name = None
    best_pipeline = None
    best_score = -1.0
    metrics_report = {}

    for name, clf in candidates.items():
        # ImbPipeline ensures SMOTE runs ONLY during .fit() on training data
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("classifier", clf),
            ]
        )

        pipeline.fit(X_train, y_train)

        # Predict fraud probability (column index 1 = probability of class 1 / fraud)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        pr_auc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        metrics_report[name] = {
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "f1": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
        }

        # Pick the best model using PR-AUC (best suited for rare/imbalanced fraud events)
        if pr_auc > best_score:
            best_score = pr_auc
            best_name = name
            best_pipeline = pipeline

    # 5. Save the trained pipeline and metadata inside artifacts/
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.joblib"
    joblib.dump(best_pipeline, model_path)

    metadata = {
        "best_model": best_name,
        "features": list(X.columns),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "merchant_categories": sorted(df["merchant_category"].unique().tolist()),
        "risk_thresholds": {
            "approve_max": 0.30,
            "flag_max": 0.75,
            "decisions": ["Approve", "Flag for Review", "Block"],
        },
        "metrics": metrics_report,
    }

    metadata_path = artifacts_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Best model: {best_name} (PR-AUC: {best_score:.4f})")
    print("Model comparison metrics:")
    for model_name, m in metrics_report.items():
        print(f"  - {model_name}: {m}")
    print(f"Saved {model_path} and {metadata_path}")


if __name__ == "__main__":
    main()