"""
AIML Assignment-2 Streamlit app
- Upload test CSV (features + Class target)
- Select one of 5 saved models
- Show metrics + confusion matrix / classification report
Models are loaded from disk — no retraining on upload.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
ART_DIR = ROOT / "artifacts"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_artifacts():
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    meta = json.loads((ART_DIR / "meta.json").read_text(encoding="utf-8"))
    train_metrics = json.loads((ART_DIR / "metrics.json").read_text(encoding="utf-8"))
    return le, models, meta, train_metrics


def compute_metrics(y_true, y_pred, y_proba, n_classes: int) -> dict:
    try:
        auc = float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=list(range(n_classes)),
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
        "F1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def main():
    st.set_page_config(page_title="Dry Bean Classifier — AIML Assignment 2", layout="wide")
    st.title("Dry Bean Classification — Assignment 2")
    st.caption("BITS Pilani WILP · AIML · 5 classifiers · metrics on uploaded test CSV (no retraining)")

    try:
        le, models, meta, train_metrics = load_artifacts()
    except Exception as e:
        st.error(f"Could not load saved models/artifacts. Run `python train_models.py` first.\n\n{e}")
        return

    feature_names = meta["feature_names"]
    target_col = meta["target_column"]
    class_names = meta["class_names"]

    with st.sidebar:
        st.header("About")
        st.write(f"**Dataset:** {meta['dataset']}")
        st.write(f"**Features:** {meta['n_features']} · **Rows (full):** {meta['n_instances']}")
        st.write(f"**Classes:** {', '.join(class_names)}")
        st.markdown("---")
        st.subheader("Training-set metrics (reference)")
        st.dataframe(pd.DataFrame(train_metrics).T, use_container_width=True)

    st.subheader("1) Upload test CSV")
    st.write(
        f"CSV must include columns: `{', '.join(feature_names)}` and target `{target_col}`."
    )
    uploaded = st.file_uploader("Upload test data CSV", type=["csv"])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Use bundled test_data.csv"):
            st.session_state["use_bundled"] = True
    with col_b:
        sample_path = ROOT / "test_data_sample.csv"
        if sample_path.exists():
            st.download_button(
                "Download sample test CSV",
                data=sample_path.read_bytes(),
                file_name="test_data_sample.csv",
                mime="text/csv",
            )

    df = None
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["use_bundled"] = False
    elif st.session_state.get("use_bundled"):
        df = pd.read_csv(ROOT / "test_data.csv")

    if df is None:
        st.info("Upload a CSV or click **Use bundled test_data.csv** to continue.")
        return

    st.write(f"Loaded **{len(df)}** rows.")
    st.dataframe(df.head(), use_container_width=True)

    missing = [c for c in feature_names + [target_col] if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    X = df[feature_names]
    y_raw = df[target_col].astype(str)
    # Map labels; drop unknown classes if any
    known = set(class_names)
    mask = y_raw.isin(known)
    if not mask.all():
        st.warning(f"Dropped {(~mask).sum()} rows with unknown Class labels.")
        X = X.loc[mask]
        y_raw = y_raw.loc[mask]
    y_true = le.transform(y_raw)

    st.subheader("2) Select model")
    model_name = st.selectbox("Classification model", list(MODEL_FILES.keys()))
    model = models[model_name]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = compute_metrics(y_true, y_pred, y_proba, n_classes=len(class_names))

    st.subheader("3) Evaluation metrics (on uploaded test data)")
    mcols = st.columns(6)
    for i, (k, v) in enumerate(metrics.items()):
        mcols[i].metric(k, v if v is not None else "N/A")

    st.dataframe(pd.DataFrame([metrics], index=[model_name]), use_container_width=True)

    st.subheader("4) Confusion matrix & classification report")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    st.pyplot(fig)
    plt.close(fig)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    with st.expander("Compare all models on this uploaded CSV"):
        rows = {}
        for name, mdl in models.items():
            pred = mdl.predict(X)
            proba = mdl.predict_proba(X)
            rows[name] = compute_metrics(y_true, pred, proba, n_classes=len(class_names))
        st.dataframe(pd.DataFrame(rows).T, use_container_width=True)


if __name__ == "__main__":
    main()
