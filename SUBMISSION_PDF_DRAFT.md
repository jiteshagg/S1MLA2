# AIML Machine Learning — Assignment 2 Submission

**Student:** Jitesh Aggarwal  
**Course:** M.Tech (AIML) · Machine Learning  
**Marks:** 15 · **Deadline:** 08-Nov-2026

---

## 1. GitHub Repository Link

https://github.com/jiteshagg/S1MLA2

Contains: `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, `model/*.joblib`, `train_models.py`, `artifacts/`.

---

## 2. Live Streamlit App Link

https://s1mla2-wwgpemq3rrfc8gdmcvjra8.streamlit.app/

---

## 3. BITS Virtual Lab Screenshot

> **TODO — insert ONE screenshot** of assignment execution on BITS Virtual Lab  
> (run training / Streamlit there and capture the screen).

---

## 4. README content (required structure)

### a. Problem statement

Build and compare five supervised classification models on a public multi-class dataset, report standard evaluation metrics, package the trained models in a Streamlit web app (upload test CSV → select model → metrics + confusion matrix), and deploy on Streamlit Community Cloud. Models are trained once offline; the app loads saved models and evaluates uploaded test data **without retraining**.

### b. Dataset description

| Item | Detail |
|------|--------|
| Name | UCI Dry Bean Dataset |
| Source | https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| Task | Multi-class classification (7 bean varieties) |
| Instances | 13,611 (≥ 500) |
| Features | 16 numeric features (≥ 12) |
| Target | `Class` ∈ {SEKER, BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SIRA} |
| Split | Stratified 80% train / 20% test (`random_state=42`) |

### c. GitHub Repository Link

https://github.com/jiteshagg/S1MLA2

### d. Models used — comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.9115 | 0.9649 | 0.9114 | 0.9115 | 0.9113 | 0.8930 |
| KNN | 0.9137 | 0.9837 | 0.9144 | 0.9137 | 0.9139 | 0.8956 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong accuracy (~92.1%) and excellent AUC (~0.993). Linear boundaries work well after standardization. |
| Decision Tree | Slightly lower accuracy (~91.2%) and AUC (~0.965). Interpretable but more variance-sensitive. |
| KNN | Competitive (~91.4%). Needs scaling; neighbourhood-based. |
| Naive Bayes | Lowest accuracy (~89.8%) but high AUC (~0.990). Feature independence assumption is imperfect. |
| Support Vector Machine | Best overall: highest Accuracy, AUC, F1, MCC. RBF kernel fits non-linear boundaries. |
| **Overall winner?** | **Support Vector Machine (RBF)**; Logistic Regression is a close second. |
