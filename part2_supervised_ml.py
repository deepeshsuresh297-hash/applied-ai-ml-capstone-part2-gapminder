"""Part 2 — Supervised Machine Learning Model: Build, Train, and Evaluate.

This script uses cleaned_data.csv from Part 1 of the Applied AI & ML Essentials
capstone. It trains and evaluates:
  1) an OLS linear regression model and Ridge regression model for life expectancy;
  2) a logistic-regression classifier for above-median life expectancy;
  3) a strongly regularised logistic-regression comparison model; and
  4) a bootstrap confidence interval for the difference in the two ROC-AUC values.

Run from the project root:
    python part2_supervised_ml.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
RIDGE_ALPHA = 1.0
BASELINE_C = 1.0
STRONG_REGULARISATION_C = 0.01
BOOTSTRAP_SAMPLES = 500

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "cleaned_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET_COLUMN = "lifeExp"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def print_heading(title: str) -> None:
    """Print a clear section heading for terminal output."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def make_preprocessor(
    categorical_columns: List[str], numeric_columns: List[str]
) -> ColumnTransformer:
    """Create an encoder fitted later on training data only.

    `iso_num` is an identifier, not a measured quantity, so it is intentionally
    handled as categorical even though it is stored as an integer in the CSV.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    drop="first",           # avoids dummy-variable redundancy within a column
                    handle_unknown="ignore", # makes a future prediction robust to an unseen category
                    sparse_output=False,
                ),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def metrics_row(
    model_name: str, y_true: pd.Series, y_pred: np.ndarray
) -> Dict[str, float | str]:
    """Return the requested regression metrics for one fitted model."""
    return {
        "model": model_name,
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def classifier_metrics(
    y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray
) -> Dict[str, float]:
    """Return common classification metrics for a binary classifier."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, probabilities),
    }


def save_table(table: pd.DataFrame, filename: str) -> None:
    """Save a generated result table to the outputs folder."""
    table.to_csv(OUTPUT_DIR / filename, index=False)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.name}. Copy cleaned_data.csv from Part 1 "
            "into this project folder before running this script."
        )

    # -------------------------------------------------------------------------
    # 1. Load the cleaned Part 1 data and define labels/features
    # -------------------------------------------------------------------------
    print_heading("1. LOAD CLEANED DATA AND DEFINE LABELS")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH.name} with shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected regression target '{TARGET_COLUMN}' was not found.")

    y_reg = df[TARGET_COLUMN].astype(float)
    median_life_expectancy = y_reg.median()
    y_clf = (y_reg > median_life_expectancy).astype(int)

    # The feature matrix contains every original non-target column.  Do not add
    # y_clf to df/X: it is derived from y_reg and would leak the target.
    X = df.drop(columns=[TARGET_COLUMN]).copy()

    # country, continent and iso_alpha are nominal strings. iso_num is a country
    # identifier and has no numerical ordering, so it is also treated as nominal.
    categorical_columns = ["country", "continent", "iso_alpha", "iso_num"]
    numeric_columns = [column for column in X.columns if column not in categorical_columns]
    X["iso_num"] = X["iso_num"].astype(str)

    print(f"Regression label y_reg: {TARGET_COLUMN} (continuous life expectancy).")
    print(
        "Classification label y_clf: 1 when lifeExp is greater than "
        f"its dataset median ({median_life_expectancy:.4f}), otherwise 0."
    )
    print("Feature matrix X columns:", X.columns.tolist())
    print("Categorical columns encoded with one-hot encoding:", categorical_columns)
    print("Numeric columns passed through before scaling:", numeric_columns)

    # -------------------------------------------------------------------------
    # 2–3. Leak-free split, one-hot encoding and StandardScaler fitting
    # -------------------------------------------------------------------------
    print_heading("2–3. LEAK-FREE TRAIN–TEST SPLIT, ENCODING AND SCALING")
    (
        X_train,
        X_test,
        y_reg_train,
        y_reg_test,
        y_clf_train,
        y_clf_test,
    ) = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )

    print(f"Train feature shape before encoding: {X_train.shape}")
    print(f"Test feature shape before encoding:  {X_test.shape}")
    print("\nClassification label counts in training set:")
    print(y_clf_train.value_counts().sort_index().rename("count"))

    # Encoding is fitted ONLY on X_train. This avoids letting the test-set
    # category vocabulary influence training preprocessing.
    encoder = make_preprocessor(categorical_columns, numeric_columns)
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    feature_names = encoder.get_feature_names_out()

    # The scaler is explicitly fitted ONLY on encoded training features, then
    # applied unchanged to encoded training and test features. The task requests
    # StandardScaler; scaling all encoded columns puts all model inputs on a
    # common scale and makes coefficient magnitudes comparable.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    print(f"Encoded/scaled train shape: {X_train_scaled.shape}")
    print(f"Encoded/scaled test shape:  {X_test_scaled.shape}")
    print(
        "Leakage check: OneHotEncoder and StandardScaler were fit on X_train only; "
        "X_test was transformed using those already-fitted objects."
    )

    # -------------------------------------------------------------------------
    # 4. Regression: OLS Linear Regression and Ridge Regression
    # -------------------------------------------------------------------------
    print_heading("4. REGRESSION MODELS — LINEAR REGRESSION AND RIDGE")
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_reg_train)
    y_pred_linear = linear_model.predict(X_test_scaled)

    ridge_model = Ridge(alpha=RIDGE_ALPHA)
    ridge_model.fit(X_train_scaled, y_reg_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)

    regression_results = pd.DataFrame(
        [
            metrics_row("Linear Regression (OLS)", y_reg_test, y_pred_linear),
            metrics_row(f"Ridge Regression (alpha={RIDGE_ALPHA})", y_reg_test, y_pred_ridge),
        ]
    )
    print("Regression comparison:")
    print(regression_results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    save_table(regression_results, "regression_model_comparison.csv")

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": linear_model.coef_,
        }
    )
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("absolute_coefficient", ascending=False)
    top_three_coefficients = coefficients.head(3).copy()

    print("\nTop three OLS coefficients by absolute value:")
    print(top_three_coefficients.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    save_table(coefficients, "linear_regression_coefficients.csv")

    # -------------------------------------------------------------------------
    # 5. Classification: baseline Logistic Regression and ROC diagnostics
    # -------------------------------------------------------------------------
    print_heading("5. CLASSIFICATION MODEL — LOGISTIC REGRESSION")
    class_counts_before = y_clf_train.value_counts().sort_index()
    smallest_class_share = class_counts_before.min() / class_counts_before.sum()

    # This dataset is balanced. The conditional code still handles an imbalance
    # in a future client dataset using class_weight='balanced' without resampling.
    if smallest_class_share < 0.35:
        class_weight_setting: str | None = "balanced"
        imbalance_decision = (
            "The minority class is below 35%, so LogisticRegression uses "
            "class_weight='balanced'. This changes the optimisation penalty, "
            "not the row count."
        )
    else:
        class_weight_setting = None
        imbalance_decision = (
            "No class has fewer than 35% of the training rows, so no resampling or "
            "class weighting is required."
        )

    # There is no resampling in this dataset, therefore the count comparison is
    # intentionally unchanged. It is printed to show that this decision is data-driven.
    class_counts_after = y_clf_train.value_counts().sort_index()
    class_balance_table = pd.DataFrame(
        {
            "class": class_counts_before.index,
            "before_handling": class_counts_before.values,
            "after_handling": class_counts_after.reindex(class_counts_before.index).values,
        }
    )
    print(imbalance_decision)
    print("\nClass-count comparison:")
    print(class_balance_table.to_string(index=False))
    save_table(class_balance_table, "class_balance_comparison.csv")

    logistic_model = LogisticRegression(
        C=BASELINE_C,
        max_iter=1000,
        solver="lbfgs",
        class_weight=class_weight_setting,
        random_state=RANDOM_STATE,
    )
    logistic_model.fit(X_train_scaled, y_clf_train)

    probabilities_c1 = logistic_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_clf = logistic_model.predict(X_test_scaled)
    baseline_metrics = classifier_metrics(y_clf_test, y_pred_clf, probabilities_c1)
    baseline_confusion = confusion_matrix(y_clf_test, y_pred_clf)

    print("\nConfusion matrix [rows=true class, columns=predicted class]:")
    print(baseline_confusion)
    print("\nClassification report:")
    print(classification_report(y_clf_test, y_pred_clf, digits=4, zero_division=0))
    print("Baseline logistic-regression metrics:")
    for metric_name, metric_value in baseline_metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")

    # Save a visual confusion matrix generated directly by this code.
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=baseline_confusion, display_labels=[0, 1])
    display.plot(ax=ax, colorbar=False)
    ax.set_title("Baseline Logistic Regression — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_logistic_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Required ROC curve and AUC annotation.
    false_positive_rate, true_positive_rate, _ = roc_curve(y_clf_test, probabilities_c1)
    plt.figure(figsize=(7, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {baseline_metrics['AUC']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="No-skill reference")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Baseline Logistic Regression")
    plt.legend(loc="lower right")
    plt.text(0.58, 0.14, f"AUC = {baseline_metrics['AUC']:.4f}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_logistic_roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 5b. Decision-threshold sensitivity table
    # -------------------------------------------------------------------------
    print_heading("5b. DECISION-THRESHOLD SENSITIVITY")
    threshold_rows: List[Dict[str, float]] = []
    for threshold in np.arange(0.30, 0.71, 0.10):
        threshold = round(float(threshold), 2)
        threshold_predictions = (probabilities_c1 >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_clf_test, threshold_predictions, zero_division=0),
                "recall": recall_score(y_clf_test, threshold_predictions, zero_division=0),
                "F1": f1_score(y_clf_test, threshold_predictions, zero_division=0),
            }
        )

    threshold_table = pd.DataFrame(threshold_rows)
    best_threshold_row = threshold_table.loc[threshold_table["F1"].idxmax()]
    print(threshold_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(
        f"\nF1 is maximised at threshold {best_threshold_row['threshold']:.2f} "
        f"with F1 = {best_threshold_row['F1']:.6f}."
    )
    save_table(threshold_table, "threshold_sensitivity.csv")

    # -------------------------------------------------------------------------
    # 6. Stronger L2 regularisation experiment
    # -------------------------------------------------------------------------
    print_heading("6. LOGISTIC-REGRESSION REGULARISATION EXPERIMENT")
    strongly_regularised_model = LogisticRegression(
        C=STRONG_REGULARISATION_C,
        max_iter=1000,
        solver="lbfgs",
        class_weight=class_weight_setting,
        random_state=RANDOM_STATE,
    )
    strongly_regularised_model.fit(X_train_scaled, y_clf_train)

    probabilities_c001 = strongly_regularised_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_c001 = strongly_regularised_model.predict(X_test_scaled)
    strong_metrics = classifier_metrics(y_clf_test, y_pred_c001, probabilities_c001)

    regularisation_comparison = pd.DataFrame(
        [
            {
                "model": f"Logistic Regression (C={BASELINE_C})",
                "precision": baseline_metrics["precision"],
                "recall": baseline_metrics["recall"],
                "F1": baseline_metrics["F1"],
                "AUC": baseline_metrics["AUC"],
            },
            {
                "model": f"Logistic Regression (C={STRONG_REGULARISATION_C})",
                "precision": strong_metrics["precision"],
                "recall": strong_metrics["recall"],
                "F1": strong_metrics["F1"],
                "AUC": strong_metrics["AUC"],
            },
        ]
    )
    print(regularisation_comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    save_table(regularisation_comparison, "logistic_regularisation_comparison.csv")

    # -------------------------------------------------------------------------
    # 7. Bootstrap confidence interval for the difference in AUC
    # -------------------------------------------------------------------------
    print_heading("7. BOOTSTRAP CONFIDENCE INTERVAL FOR AUC DIFFERENCE")
    # The task calls for np.random.choice; setting a seed makes the 500 sampled
    # AUC differences reproducible.
    np.random.seed(RANDOM_STATE)
    y_test_array = y_clf_test.to_numpy()
    auc_differences: List[float] = []

    # Resample until 500 valid samples are collected. A bootstrap sample with
    # only one class would make ROC-AUC undefined, so it is redrawn if it occurs.
    while len(auc_differences) < BOOTSTRAP_SAMPLES:
        sampled_indices = np.random.choice(
            len(y_test_array), size=len(y_test_array), replace=True
        )
        y_bootstrap = y_test_array[sampled_indices]
        if np.unique(y_bootstrap).size < 2:
            continue

        auc_c1 = roc_auc_score(y_bootstrap, probabilities_c1[sampled_indices])
        auc_c001 = roc_auc_score(y_bootstrap, probabilities_c001[sampled_indices])
        auc_differences.append(auc_c1 - auc_c001)

    auc_differences_array = np.asarray(auc_differences)
    mean_auc_difference = auc_differences_array.mean()
    lower_ci, upper_ci = np.percentile(auc_differences_array, [2.5, 97.5])

    bootstrap_summary = pd.DataFrame(
        [
            {
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "mean_auc_difference_C1_minus_C001": mean_auc_difference,
                "ci_2_5_percent": lower_ci,
                "ci_97_5_percent": upper_ci,
                "ci_excludes_zero": bool(lower_ci > 0 or upper_ci < 0),
            }
        ]
    )
    print(bootstrap_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    save_table(bootstrap_summary, "bootstrap_auc_difference_summary.csv")

    # A compact text summary is useful when reviewing the repository without
    # scrolling through a terminal log.
    summary_lines = [
        "Part 2 generated results summary",
        f"Regression target: {TARGET_COLUMN}",
        f"Classification median threshold: {median_life_expectancy:.4f}",
        "",
        "Regression model comparison:",
        regression_results.to_string(index=False),
        "",
        "Top three OLS coefficients by absolute value:",
        top_three_coefficients.to_string(index=False),
        "",
        "Baseline logistic metrics:",
        pd.DataFrame([baseline_metrics]).to_string(index=False),
        "",
        "Best F1 threshold:",
        best_threshold_row.to_frame().T.to_string(index=False),
        "",
        "Regularisation comparison:",
        regularisation_comparison.to_string(index=False),
        "",
        "Bootstrap AUC-difference summary:",
        bootstrap_summary.to_string(index=False),
    ]
    (OUTPUT_DIR / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print_heading("COMPLETE")
    print("All required tables and plots were generated successfully.")
    print(f"Generated files are available in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
