# Part 2 — Supervised Machine Learning Model: Build, Train, and Evaluate

## Project overview

This repository is the Part 2 submission for the **Applied AI & ML Essentials** capstone. It loads the clean Gapminder country-year dataset produced in Part 1 and builds two supervised-learning tasks:

1. **Regression:** predict continuous life expectancy (`lifeExp`).
2. **Binary classification:** predict whether life expectancy is above the dataset median.

The project is fully reproducible: `part2_supervised_ml.py` performs all preprocessing, model fitting, evaluation, plot generation, threshold analysis, and bootstrap confidence-interval calculations. No API keys or secrets are required.

## Dataset and labels

`cleaned_data.csv` is the clean dataset created in Part 1. It contains 1,704 country-year observations and these original fields:

| Column | Role in Part 2 |
|---|---|
| `country` | Categorical predictor |
| `continent` | Categorical predictor |
| `year` | Numeric predictor |
| `lifeExp` | Regression target (`y_reg`) |
| `pop` | Numeric predictor |
| `gdpPercap` | Numeric predictor |
| `iso_alpha` | Categorical country identifier predictor |
| `iso_num` | Categorical country identifier predictor |

### Target definitions

- **Regression label `y_reg`:** `lifeExp`, a continuous value measured in years.
- **Classification label `y_clf`:** `(lifeExp > lifeExp.median()).astype(int)`.

The median life expectancy is **60.7125 years**. Therefore, class `1` means a country-year observation has life expectancy above 60.7125, while class `0` means life expectancy is at or below the median.

`X` contains every original column except `lifeExp`. The derived classification label is never added to `X`, because it is calculated directly from the regression target and would leak the answer.

## Repository contents

```text
part2_gapminder/
├── cleaned_data.csv
├── part2_supervised_ml.py
├── requirements.txt
├── .gitignore
├── README.md
└── outputs/                         # generated directly by the Python script
    ├── 01_logistic_confusion_matrix.png
    ├── 02_logistic_roc_curve.png
    ├── bootstrap_auc_difference_summary.csv
    ├── class_balance_comparison.csv
    ├── linear_regression_coefficients.csv
    ├── logistic_regularisation_comparison.csv
    ├── regression_model_comparison.csv
    ├── run_summary.txt
    └── threshold_sensitivity.csv
```

## Installation and execution

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and run the complete pipeline:

```bash
pip install -r requirements.txt
python part2_supervised_ml.py
```

The script runs from top to bottom without manual inputs. It prints all required metrics and tables in the terminal, then writes the ROC plot, confusion-matrix plot, result tables, and a concise run summary into `outputs/`.

## Preprocessing and leakage prevention

### Categorical encoding

The categorical columns are `country`, `continent`, `iso_alpha`, and `iso_num`.

- `country`, `continent`, and `iso_alpha` are nominal categories with no natural numerical order.
- `iso_num` is stored as an integer in the CSV, but it is a country identifier rather than a measured quantity. It is therefore treated as a nominal category as well.

All four fields are encoded with `OneHotEncoder(drop='first', handle_unknown='ignore')`. Dropping the first dummy category avoids redundant dummy columns within each original variable. One-hot encoding is preferable to label encoding here because an artificial code such as Africa = 0, Asia = 1, Europe = 2 would wrongly imply that Europe is numerically “twice” Asia or ordered above it.

### Train-test split and scaling

The code uses:

```python
train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
```

This makes an 80% training set and 20% test set while preserving the balanced class proportion. The one-hot encoder is fitted on `X_train` only. Then `StandardScaler` is fitted on the resulting encoded training matrix only:

```python
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

scaler.fit(X_train_encoded)
X_train_scaled = scaler.transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)
```

Fitting a scaler on the full dataset would be data leakage: the training process would indirectly receive the test-set means and standard deviations. The evaluation would then be overly optimistic because the test set would no longer be completely unseen.

The encoded feature matrix has 428 columns after one-hot encoding and has 1,363 training rows and 341 test rows.

## Regression model results

Two models predict `lifeExp` from the same scaled train/test features.

| Model | Test MSE | Test R² |
|---|---:|---:|
| Linear Regression (OLS) | 13.193297 | 0.920854 |
| Ridge Regression (`alpha=1.0`) | 13.192262 | 0.920860 |

Both models explain about **92.1%** of the variance in life expectancy on the held-out test set. Ridge produces a very small improvement in MSE and R² here.

### Linear-regression coefficient interpretation

The table below lists the three largest absolute OLS coefficients. All model inputs were standardized, so each coefficient represents the predicted change in life expectancy for a one-standard-deviation increase in that transformed feature, holding the other transformed features fixed.

| Feature | Coefficient | Interpretation |
|---|---:|---|
| `year` | +5.590646 | A later standardized year is associated with an increase of about 5.59 predicted life-expectancy years, conditional on the other features. This captures the broad time trend in the data. |
| `pop` | +1.995010 | A one-standard-deviation increase in transformed population is associated with about 2.00 more predicted life-expectancy years, after controlling for the other included variables. This is an association, not proof that a larger population causes longer life expectancy. |
| `continent_Europe` | +1.800299 | The standardized Europe indicator is positively associated with the prediction relative to the omitted continent reference category and the other model inputs. |

A **large positive coefficient** means higher values of the standardized feature are associated with a higher prediction; a **large negative coefficient** means they are associated with a lower prediction. The country and country-code variables are identifiers and contain overlapping information, so their individual coefficients are not causal effects and should be treated cautiously.

### Why Ridge can differ from OLS

Ridge Regression adds an L2 penalty to the loss function. `alpha` controls the strength of that penalty: larger values shrink coefficients more strongly toward zero. Ridge is useful when predictors are correlated or redundant, as in this dataset where country name and country identifiers overlap. The penalty can stabilise coefficient estimates and improve generalisation, though excessive shrinkage can discard useful signal.

## Classification model results

### Class-balance check

The training labels are balanced:

| Class | Before handling | After handling |
|---:|---:|---:|
| 0 — life expectancy at/below median | 681 | 681 |
| 1 — life expectancy above median | 682 | 682 |

Neither class is below the 35% threshold. Therefore, no SMOTE or `class_weight='balanced'` intervention was applied. The script still implements a conditional `class_weight='balanced'` pathway for a future client dataset with a minority class below 35%. The before/after counts remain identical because no resampling is needed.

### Baseline Logistic Regression (`C=1.0`, `max_iter=1000`)

The baseline model is trained with `LogisticRegression`, and its probabilities are created with `predict_proba(X_test_scaled)[:, 1]`.

**Confusion matrix** — rows are true values and columns are predicted values:

| True class | Predicted 0 | Predicted 1 |
|---:|---:|---:|
| 0 | 155 | 16 |
| 1 | 7 | 163 |

| Metric | Value |
|---|---:|
| Accuracy | 0.932551 |
| Precision | 0.910615 |
| Recall | 0.958824 |
| F1-score | 0.934097 |
| ROC-AUC | 0.984211 |

The ROC curve is saved as `outputs/02_logistic_roc_curve.png` and annotates the AUC. An AUC of **0.9842** means that the model has excellent ability to rank a randomly chosen above-median observation above a randomly chosen at/below-median observation. It does not prove that the model will perform equally well on new countries not represented in the training data.

### Precision and recall

For the positive class (`y_clf = 1`):

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

Here, a false positive would label a lower-life-expectancy country-year as above-median. In a public-health prioritisation context, that could lead to a record being under-prioritised for support. Therefore, **precision** is the more important operational metric for the positive “above-median” label. Raising the decision threshold generally improves precision by being more selective, but the cost is lower recall: more truly above-median observations become false negatives.

## Decision-threshold sensitivity

The default logistic-regression threshold is 0.50. The code tests five thresholds from 0.30 to 0.70.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.30 | 0.864583 | 0.976471 | 0.917127 |
| 0.40 | 0.887097 | 0.970588 | 0.926966 |
| 0.50 | 0.910615 | 0.958824 | 0.934097 |
| 0.60 | 0.941176 | 0.941176 | **0.941176** |
| 0.70 | 0.951515 | 0.923529 | 0.937313 |

The F1-score is highest at a threshold of **0.60**. For a decision that prioritises precision, I would raise the threshold from 0.50 toward 0.70. This reduces false positives, but it also increases false negatives because some genuinely above-median observations will no longer pass the stricter decision rule.

## Logistic-regression regularisation experiment

A second logistic regression uses `C=0.01`, which applies stronger L2 regularisation than the baseline `C=1.0`. In logistic regression, `C` is the inverse of regularisation strength: a smaller `C` means stronger coefficient shrinkage.

| Model | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|
| Logistic Regression (`C=1.0`) | 0.910615 | 0.958824 | 0.934097 | 0.984211 |
| Logistic Regression (`C=0.01`) | 0.905028 | 0.952941 | 0.928367 | 0.969281 |

Stronger regularisation slightly worsened precision, recall, F1, and AUC on this test set. It may have shrunk useful country, time, and economic signal too much. The result does not mean regularisation is generally harmful; it means this particular amount of shrinkage was not the best setting for this split and feature representation.

## Bootstrap confidence interval for AUC difference

The script draws **500** bootstrap test-set samples with replacement using `np.random.choice`. For every sample it computes:

\[
\text{AUC difference} = \text{AUC}_{C=1.0} - \text{AUC}_{C=0.01}
\]

| Bootstrap samples | Mean AUC difference | 2.5th percentile | 97.5th percentile |
|---:|---:|---:|---:|
| 500 | 0.014832 | 0.006832 | 0.024837 |

The 95% bootstrap interval **[0.006832, 0.024837] excludes zero**. On this test set, the baseline `C=1.0` model’s AUC advantage over the `C=0.01` model appears consistently positive across the bootstrap resamples.

## Modelling caveat

This assessment requires a random row-level train-test split and all non-target variables in `X`. As a consequence, train and test contain records from the same countries at different years, and country/identifier features carry strong information. The reported metrics therefore evaluate the specified task correctly, but may be optimistic for a deployment that must predict life expectancy for completely unseen countries. A future production evaluation should use a group-aware split by country and reconsider whether country identifiers are appropriate features.

## Reproducibility checklist

- `cleaned_data.csv` is included and loaded directly by the script.
- `part2_supervised_ml.py` contains all preprocessing, training, evaluation, ROC plotting, threshold analysis, regularisation comparison, and bootstrap logic.
- The encoder and scaler fit only on training data.
- The script uses `predict_proba()` for probabilities, `roc_auc_score()` for AUC, and `np.random.choice()` for bootstrap sampling.
- All results are generated directly by code and saved under `outputs/`.
