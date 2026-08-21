# Comparative Analysis of Machine Learning Algorithms for Diabetes Risk Prediction Using the DiaBD Bangladesh Dataset

## Objective

To compare the performance of four machine learning algorithms — **Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, and XGBoost** — for binary classification of diabetes risk (Diabetic / Non-Diabetic) using the DiaBD Bangladesh clinical dataset, with particular attention to how each algorithm handles severe class imbalance.

## Dataset

**DiaBD: A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh** (`data/diabd.csv`).

- **Size (raw):** 5,288 patient records × 15 columns
- **Size (after cleaning):** 5,281 records (7 rows removed — see "Preprocessing" below)
- **Target variable:** `diabetic` (Yes / No)
- **Class distribution (raw):** No = 4,946 (93.53%), Yes = 342 (6.47%) — a severe ~14.5 : 1 imbalance
- **Numerical features (8):** age, pulse_rate, systolic_bp, diastolic_bp, glucose (mmol/L), height (m), weight (kg), bmi
- **Categorical features (6):** gender, family_diabetes, hypertensive, family_hypertension, cardiovascular_disease, stroke
- No missing values and no duplicate rows were found in the raw data.

## Algorithms Compared

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Random Forest
4. XGBoost

## Project Structure

```text
ML-assignment/
│
├── data/
│   └── diabd.csv                  # DiaBD dataset (copied from the downloaded folder)
├── notebooks/
│   └── diabetes_analysis.ipynb    # Full narrative notebook (EDA -> final model)
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Paths, random seed, column lists
│   ├── data_preprocessing.py      # Load, clean, encode, split, ColumnTransformer
│   ├── eda.py                     # Exploratory analysis plotting functions
│   ├── train_models.py            # Model pipelines, SMOTE pipelines, tuning
│   └── evaluate_models.py         # Metrics, confusion matrices, ROC, feature importance
├── outputs/
│   ├── figures/                   # All saved plots (EDA + evaluation)
│   ├── tables/                    # All result tables (CSV)
│   └── models/                    # Saved trained models (joblib)
├── report/
│   └── final_report.md            # Full academic report
├── requirements.txt
├── README.md
└── main.py                        # Runs the entire pipeline end-to-end
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run

**Full pipeline (recommended — reproduces every table and figure in `outputs/`):**

```bash
python main.py
```

This runs on a normal laptop in under 30 seconds (measured: ~19s on the development machine) and prints every stage of the analysis to the terminal.

**Notebook (for the full narrative walkthrough):**

```bash
jupyter notebook notebooks/diabetes_analysis.ipynb
```

## Evaluation Metrics

Because ~94% of patients are non-diabetic, **accuracy alone is misleading** — a model that always predicts "non-diabetic" would still score ~93.5% accuracy while catching zero diabetic patients. Every model is therefore evaluated on:

- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix and full classification report
- Special attention to **recall on the diabetic class**, since in a screening context a missed diabetic patient (false negative) is more costly than a false alarm (false positive)

## Main Findings

*(from the actual run recorded in `outputs/tables/`; see `report/final_report.md` for full discussion)*

- **All four baseline models score 93–94% accuracy but catch very few diabetic patients** — baseline diabetic-class recall ranged from 4.4% (KNN) to 22.1% (XGBoost), confirming that accuracy is a misleading metric on this dataset.
- **Class weighting and SMOTE both substantially raised diabetic-class recall** at the cost of precision/accuracy — e.g. class-weighted Logistic Regression raised recall from 14.7% to 63.2% (F1 rose from 0.225 to 0.350).
- **After hyperparameter tuning (5-fold CV, scoring = F1),** cross-validation itself selected `class_weight="balanced"` for Logistic Regression and Random Forest, and a high `scale_pos_weight` for XGBoost — confirming imbalance handling genuinely helps on this dataset rather than being an assumption we imposed.
- **Final tuned test-set results:** XGBoost achieved the best F1-score (0.388) and a strong ROC-AUC (0.856); Random Forest achieved the best ROC-AUC (0.863); Logistic Regression achieved by far the best recall (63.2%) but at low precision (24.0%).
- **`hypertensive` status and `glucose` were consistently the two strongest predictors** across all three interpretable models (Logistic Regression coefficients, Random Forest importances, XGBoost importances) — medically plausible given the known clinical association between hypertension and diabetes, and confirmed directly in the data (31.4% diabetes prevalence among hypertensive patients vs. 3.4% among non-hypertensive patients).
- **Overall recommended model: XGBoost**, based on the best balance of precision and recall (F1) combined with near-best ROC-AUC — see `report/final_report.md` Section 12 for the full trade-off discussion.

The best model and every tuned pipeline are saved to `outputs/models/` via `joblib`.
