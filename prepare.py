"""
FROZEN -- Do not modify this file.
Data loading, train/val/test split, evaluation metric, logging, and plotting.

Dataset: mal_manga_adaptation_dataset.csv
Task:    Binary classification — predict adapted_to_anime (1=yes, 0=no)
Metric:  ROC-AUC on validation set (higher is better)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import csv
import os

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED   = 42
VAL_FRACTION  = 0.15   # 15% validation (used during agent loop)
TEST_FRACTION = 0.15   # 15% test (LOCKED — never touched during training/tuning)
RESULTS_FILE  = "results.tsv"
DATA_FILE     = "mal_manga_adaptation_dataset.csv"

# ── Features ───────────────────────────────────────────────
# Only adaptation-independent features — no score, rank, popularity, etc.
# (those are consequences of adaptation, not predictors)
CATEGORICAL_FEATURES = ["media_type", "status", "nsfw"]

NUMERIC_FEATURES = [
    "num_volumes",
    "num_chapters",
    "start_year",
    "run_years",
    "in_jump_magazine",
    "in_major_magazine",
]

GENRE_FEATURES = [
    "genre_action", "genre_adventure", "genre_comedy", "genre_drama",
    "genre_fantasy", "genre_horror", "genre_mystery", "genre_romance",
    "genre_scifi", "genre_slice_of_life", "genre_sports", "genre_supernatural",
    "genre_thriller", "genre_shounen", "genre_shoujo", "genre_seinen",
    "genre_josei", "genre_ecchi",
]

TARGET = "adapted_to_anime"

# ── Data ───────────────────────────────────────────────────
def load_data():
    """
    Load MAL manga dataset and return train/val/test splits.

    Splits (stratified by target):
      - 70% train
      - 15% val   (used by agent loop for keep/discard decisions)
      - 15% test  (LOCKED — report final numbers on this only at the end)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: '{DATA_FILE}'\n"
            "Run the data collection notebook first to generate this file."
        )

    df = pd.read_csv(DATA_FILE)

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)

    # Collect all feature columns (numeric + genre + one-hot encoded categoricals)
    cat_encoded_cols = [
        c for c in df_encoded.columns
        if any(c.startswith(cat + "_") for cat in CATEGORICAL_FEATURES)
    ]
    feature_cols = NUMERIC_FEATURES + GENRE_FEATURES + cat_encoded_cols

    # Only keep columns that actually exist in the dataframe
    feature_cols = [c for c in feature_cols if c in df_encoded.columns]

    X = df_encoded[feature_cols].copy()
    y = df_encoded[TARGET].copy()

    # Fill missing numeric values with median
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(X[NUMERIC_FEATURES].median())
    X = X.fillna(0)

    # First split: carve out locked test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Second split: train vs val from remaining 85%
    val_adjusted = VAL_FRACTION / (1 - TEST_FRACTION)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_adjusted,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"Dataset loaded: {len(df)} total rows")
    print(f"  Train: {len(X_train)} rows  ({len(X_train)/len(df)*100:.0f}%)")
    print(f"  Val:   {len(X_val)} rows   ({len(X_val)/len(df)*100:.0f}%)")
    print(f"  Test:  {len(X_test)} rows   ({len(X_test)/len(df)*100:.0f}%)  [LOCKED]")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class balance (train): {y_train.mean()*100:.1f}% adapted")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# ── Evaluation (frozen metric) ─────────────────────────────
def evaluate(model, X_val, y_val, verbose=False):
    """
    Primary metric: ROC-AUC (threshold-free, handles imbalance well).
    Also computes F1 and accuracy for reference.

    Returns: (val_auc, val_f1)
    """
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred       = model.predict(X_val)

    auc = float(roc_auc_score(y_val, y_pred_proba))
    f1  = float(f1_score(y_val, y_pred, zero_division=0))
    acc = float(accuracy_score(y_val, y_pred))

    if verbose:
        print(classification_report(y_val, y_pred, target_names=["not adapted", "adapted"]))

    return auc, f1, acc


# ── Logging ────────────────────────────────────────────────
def log_result(experiment_id, val_auc, val_f1, val_acc, status, description, runtime_s):
    """Append one row to results.tsv."""
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow([
                "experiment", "val_auc", "val_f1", "val_acc",
                "status", "runtime_s", "description"
            ])
        writer.writerow([
            experiment_id,
            f"{val_auc:.6f}",
            f"{val_f1:.6f}",
            f"{val_acc:.6f}",
            status,
            f"{runtime_s:.2f}",
            description,
        ])


# ── Plotting ───────────────────────────────────────────────
def plot_results(save_path="performance.png"):
    """Plot val AUC and F1 over experiments from results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first.")
        return

    experiments, aucs, f1s, statuses, descriptions, runtimes = [], [], [], [], [], []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experiments.append(row["experiment"])
            aucs.append(float(row["val_auc"]))
            f1s.append(float(row["val_f1"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])
            runtimes.append(float(row.get("runtime_s", 0)))

    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors    = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ── Top: AUC ──
    ax1.scatter(range(len(aucs)), aucs, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax1.plot(range(len(aucs)), aucs, "k--", alpha=0.2, zorder=2)

    best_auc = []
    current_best = 0.0
    for a in aucs:
        current_best = max(current_best, a)
        best_auc.append(current_best)
    ax1.plot(range(len(aucs)), best_auc, color="#2ecc71", linewidth=2.5, label="Best so far")
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Random baseline (0.5)")

    ax1.set_ylim(max(0, min(aucs) - 0.05), min(1.0, max(aucs) + 0.05))
    ax1.set_ylabel("Validation ROC-AUC (higher is better)", fontsize=11)
    ax1.set_title("AutoResearch: Manga → Anime Adaptation Classifier", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # ── Bottom: F1 ──
    ax2.scatter(range(len(f1s)), f1s, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax2.plot(range(len(f1s)), f1s, "k--", alpha=0.2, zorder=2)

    best_f1 = []
    current_best_f1 = 0.0
    for f in f1s:
        current_best_f1 = max(current_best_f1, f)
        best_f1.append(current_best_f1)
    ax2.plot(range(len(f1s)), best_f1, color="#2ecc71", linewidth=2.5, label="Best so far")

    ax2.set_ylim(max(0, min(f1s) - 0.05), min(1.0, max(f1s) + 0.05))
    ax2.set_ylabel("Validation F1 Score (higher is better)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    short_labels = [d[:22] + ".." if len(d) > 24 else d for d in descriptions]
    ax2.set_xticks(range(len(aucs)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("Experiment #", fontsize=11)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="keep (improved)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="discard (regressed)"),
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Best so far"),
        Line2D([0], [0], color="gray", linestyle=":", linewidth=1, label="Random baseline"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_results()
