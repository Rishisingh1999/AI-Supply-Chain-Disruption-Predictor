#!/usr/bin/env python3
"""
AI Supply Chain Disruption Predictor - Reproducible Experiment
Target: Disruption_Occurred (binary)
Dataset: Global Supply Chain Risk and Logistics 2024-2026 (Kaggle)
"""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score
)

SEED = 42
np.random.seed(SEED)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
DATA_PATH = Path("global_supply_chain_risk_2026.csv")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def eval_at_threshold(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist()
    }

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    file_hash = sha256_file(DATA_PATH)
    print(f"Shape: {df.shape}, SHA256: {file_hash}")
    print("Target prevalence:\n", df["Disruption_Occurred"].value_counts(normalize=True))

    # Features available at prediction time (no leakage assumed)
    numeric_features = [
        "Distance_km", "Weight_MT", "Fuel_Price_Index",
        "Geopolitical_Risk_Score", "Carrier_Reliability_Score", "Lead_Time_Days"
    ]
    categorical_features = [
        "Transport_Mode", "Product_Category", "Weather_Condition",
        "Origin_Port", "Destination_Port"
    ]

    X = df[numeric_features + categorical_features]
    y = df["Disruption_Occurred"].values

    # Stratified split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=SEED
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ])

    candidates = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=SEED
        )
    }

    val_results = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        val_results[name] = {
            "val_auc": float(roc_auc_score(y_val, proba)),
            "val_ap": float(average_precision_score(y_val, proba))
        }
        print(f"{name}: val AUC={val_results[name]['val_auc']:.4f}, AP={val_results[name]['val_ap']:.4f}")

    # Select best by val AUC
    best_name = max(val_results, key=lambda k: val_results[k]["val_auc"])
    print(f"Selected model: {best_name}")

    # Refit on train+val
    final_pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", candidates[best_name])
    ])
    final_pipe.fit(X_trainval, y_trainval)

    test_proba = final_pipe.predict_proba(X_test)[:, 1]
    metrics = eval_at_threshold(y_test, test_proba, 0.5)
    print("\n=== Test metrics ===")
    print(json.dumps(metrics, indent=2))

    # Save predictions
    pd.DataFrame({
        "y_true": y_test,
        "y_proba": test_proba,
        "y_pred": (test_proba >= 0.5).astype(int)
    }).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # EDA target
    plt.figure(figsize=(5, 4))
    df["Disruption_Occurred"].value_counts().plot(kind="bar", color=["steelblue", "salmon"])
    plt.title("Target Distribution: Disruption_Occurred")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_target.png", dpi=120)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Test) — {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=120)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_test, test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP = {metrics['average_precision']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Test) — {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curve.png", dpi=120)
    plt.close()

    # Confusion
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix (Test, thr=0.5)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    report = {
        "target_definition": "Disruption_Occurred (binary 1 = disruption occurred)",
        "dataset_source": "https://www.kaggle.com/datasets/nudratabbas/global-supply-chain-risk-and-logistics-2024-2026",
        "dataset_file": "global_supply_chain_risk_2026.csv",
        "dataset_rows": int(df.shape[0]),
        "dataset_cols": int(df.shape[1]),
        "target_prevalence": float(y.mean()),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "split_method": "stratified train_test_split (80/20), then 80/20 train/val for model selection",
        "models_compared": val_results,
        "selected_model": best_name,
        "test_metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "file_sha256": file_hash,
        "run_date": datetime.now(timezone.utc).isoformat(),
        "seed": SEED
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved all artifacts to", OUT_DIR)

if __name__ == "__main__":
    main()
