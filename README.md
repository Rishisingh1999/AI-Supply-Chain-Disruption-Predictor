# AI Supply Chain Disruption Predictor

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)

Binary classification prototype that predicts whether a shipment will experience a disruption, using the public Kaggle dataset *Global Supply Chain Risk and Logistics 2024-2026*.

> This is an educational portfolio project. Results are derived solely from the public dataset; they do not constitute operational recommendations or claims of cost savings.

---

## 1. Project Purpose

Supply-chain disruptions are expensive. This project builds a transparent baseline that:

- Predicts the binary outcome `Disruption_Occurred`.
- Uses only features that would realistically be known before or at the time of shipment planning.
- Compares a simple logistic regression against tree-based models and reports honest held-out metrics.

---

## 2. Dataset Source & Attribution

| Item | Value |
|------|-------|
| **Source** | [Kaggle – Global Supply Chain Risk and Logistics 2024-2026](https://www.kaggle.com/datasets/nudratabbas/global-supply-chain-risk-and-logistics-2024-2026) |
| **File used** | `global_supply_chain_risk_2026.csv` |
| **Rows / Columns** | 5,000 × 14 |
| **Target prevalence** | 61.26 % positive (`Disruption_Occurred = 1`) |
| **SHA-256** | `93ac93e4c53deea4c6dc7ca937326a2f1f839543c4ce0fa5fcbc300da8a03c73` |
| **Access date** | 2026-10-27 |

**Target definition**  
`Disruption_Occurred` is already a binary label (1 = disruption occurred). No re-labeling was performed.

**Feature availability note**  
All numeric and categorical features used (distance, weight, fuel-price index, geopolitical risk score, carrier reliability, lead time, transport mode, product category, weather, origin/destination ports) are treated as known at planning time. Identifiers (`Shipment_ID`) and the raw date string are excluded from the model. The exact measurement timing of the risk scores and weather labels is not fully documented by the dataset publisher; residual uncertainty remains.

---

## 3. Data Cleaning & Feature Engineering

- No missing values in the selected columns.
- Numeric features are standardized.
- Categorical features are one-hot encoded (`handle_unknown="ignore"`).
- No target-derived or post-disruption fields are present in the feature set.

---

## 4. Model & Validation Methodology

- **Split**: Stratified 80/20 train+val / test, then another stratified 80/20 inside the train+val portion for model selection. (Note: this is a random stratified split, not a temporal split.)
- **Candidates evaluated on validation AUC**:
  - Logistic Regression (class_weight="balanced")
  - Random Forest
  - Gradient Boosting
- **Selected model**: Logistic Regression (highest validation AUC).
- Final model is refit on the full train+val set and evaluated once on the held-out test set.
- Metrics: ROC-AUC, Average Precision, Precision, Recall, F1, Confusion Matrix.

---

## 5. Actual Test-Set Results (run 2026-10-27)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.828** |
| Average Precision | **0.902** |
| Precision | 0.859 |
| Recall | 0.674 |
| F1 | 0.755 |

**Confusion matrix (thr = 0.5)**

```
[[319  68]
 [200 413]]
```

Validation-set comparison (used only for model selection):

| Model | Val AUC | Val AP |
|-------|---------|--------|
| LogisticRegression | 0.827 | 0.899 |
| RandomForest | 0.819 | 0.893 |
| GradientBoosting | 0.814 | 0.892 |

All figures are produced by a fresh execution of `run_experiment.py`. After running the script, the machine-readable report appears in `outputs/metrics.json`.

---

## 6. Generated Artifacts

Running `python run_experiment.py` creates an `outputs/` directory (not committed to Git) containing:

```
outputs/
├── metrics.json
├── test_predictions.csv
├── eda_target.png
├── roc_curve.png
├── pr_curve.png
└── confusion_matrix.png
```

---

## 7. Setup & Reproduction

```bash
# 1. Clone
git clone https://github.com/Rishisingh1999/AI-Supply-Chain-Disruption-Predictor.git
cd AI-Supply-Chain-Disruption-Predictor

# 2. Install
pip install -r requirements.txt

# 3. Download the dataset
kaggle datasets download -d nudratabbas/global-supply-chain-risk-and-logistics-2024-2026 -p data --unzip
# Place global_supply_chain_risk_2026.csv in the project root or data/

# 4. Run
python run_experiment.py
```

---

## 8. Limitations & Known Risks

- The dataset is relatively small (5 k rows) and the positive class is majority; performance should be interpreted with caution.
- Feature-to-target relationships may contain synthetic or simplified patterns; real-world logistics data is noisier.
- No time-series / sequential validation was performed (dates exist but the current split is stratified random).
- Geopolitical and weather features are provided as static scores; their real-time availability and measurement methodology are unknown.
- Results must not be interpreted as causal claims or as guarantees of cost reduction.

---

## 9. Project Structure

```
AI-Supply-Chain-Disruption-Predictor/
├── run_experiment.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/                 # place CSV here (git-ignored)
└── outputs/              # generated locally by the experiment (not committed)
```

---

## License & Attribution

Educational use. Please attribute the original Kaggle dataset author.
