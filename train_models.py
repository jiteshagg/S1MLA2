"""
AIML ZG565 — Assignment 2
Train 5 classifiers on UCI Dry Bean dataset and save artifacts for Streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_XLSX = ROOT / "data" / "raw" / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx"
MODEL_DIR = ROOT / "model"
ART_DIR = ROOT / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_excel(DATA_XLSX)
    target_col = "Class"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].astype(str).copy()
    return X, y


def metrics_dict(y_true, y_pred, y_proba, labels) -> dict:
    # Multiclass AUC: one-vs-rest, weighted by support
    try:
        auc = float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=labels,
            )
        )
    except Exception:
        auc = float("nan")

    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(auc, 4) if auc == auc else None,
        "Precision": round(
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "Recall": round(
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "F1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def build_models() -> dict:
    # Scale continuous features for LR / KNN / SVM; tree & NB get raw (tree) or scaled (NB ok either way)
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=12, min_samples_leaf=5
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    print(f"Loaded Dry Bean: {X.shape[0]} rows, {X.shape[1]} features, classes={y.nunique()}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    # Persist test CSV with original class labels for Streamlit uploads
    test_df = X_test.copy()
    test_df["Class"] = le.inverse_transform(y_test)
    test_path = ROOT / "test_data.csv"
    test_df.to_csv(test_path, index=False)
    print(f"Wrote {test_path} ({len(test_df)} rows)")

    # Also keep a small sample for quick demos
    test_df.sample(n=min(200, len(test_df)), random_state=RANDOM_STATE).to_csv(
        ROOT / "test_data_sample.csv", index=False
    )

    models = build_models()
    all_metrics = {}
    reports = {}

    for name, model in models.items():
        print(f"\nTraining: {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            # Should not happen with our set
            y_proba = np.eye(len(class_names))[y_pred]

        m = metrics_dict(y_test, y_pred, y_proba, labels=list(range(len(class_names))))
        all_metrics[name] = m
        print(m)

        cm = confusion_matrix(y_test, y_pred).tolist()
        reports[name] = {
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=class_names,
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": cm,
        }

        safe = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"{safe}.joblib")

    # Shared artifacts for the app
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
    meta = {
        "dataset": "UCI Dry Bean Dataset",
        "n_features": int(X.shape[1]),
        "n_instances": int(X.shape[0]),
        "feature_names": list(X.columns),
        "target_column": "Class",
        "class_names": class_names,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": all_metrics,
    }
    (ART_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (ART_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (ART_DIR / "reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    # Pretty comparison table
    table = pd.DataFrame(all_metrics).T
    table.index.name = "ML Model Name"
    table.to_csv(ART_DIR / "comparison_table.csv")
    print("\n=== Comparison table ===")
    print(table.to_string())
    print(f"\nModels saved under: {MODEL_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
