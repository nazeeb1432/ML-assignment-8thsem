# Comparative Analysis of Machine Learning Algorithms for Diabetes Risk Prediction Using the DiaBD Bangladesh Dataset

## Abstract

Diabetes is a growing public health concern in Bangladesh, and early risk identification can help direct patients toward diagnostic testing and lifestyle intervention. This project compares four machine learning algorithms — Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, and XGBoost — for binary classification of diabetes status (Diabetic / Non-Diabetic) using the DiaBD Bangladesh clinical dataset (5,288 raw records, 14 clinical/demographic features). The dataset exhibits severe class imbalance (6.47% diabetic). All models were evaluated with a stratified 80/20 train-test split, and compared across accuracy, precision, recall, F1-score, and ROC-AUC — with particular attention to diabetic-class recall, since accuracy alone was found to be misleading (baseline models scored 93.8–94.0% accuracy while missing 69–85% of diabetic patients). Two imbalance-handling strategies (class weighting and SMOTE) were compared against the imbalanced baseline, and all four algorithms were then tuned via 5-fold cross-validated grid/randomized search optimizing F1-score. After tuning, Random Forest achieved the best F1-score (0.412), XGBoost achieved the best ROC-AUC (0.881), and Logistic Regression achieved by far the highest recall (66.2%) at the cost of precision. `hypertensive` status was consistently the most influential predictor across all three interpretable models, with `glucose` a close second for the two tree-based models specifically. The results show that class-imbalance handling is essential on this dataset (though not uniformly beneficial — plain class weighting slightly hurt Random Forest, and hyperparameter tuning slightly hurt XGBoost's and KNN's test-set generalization on some metrics) and that model choice should depend on the intended use case (screening vs. confirmatory triage) rather than accuracy alone.

## 1. Introduction

Diabetes mellitus is a chronic metabolic condition and a major cause of morbidity worldwide. In Bangladesh, rising rates of diabetes combined with limited access to routine clinical screening make early, low-cost risk identification particularly valuable — a machine learning model trained on easily-collected clinical measurements (age, blood pressure, pulse rate, glucose, BMI, and family/medical history flags) could, in principle, help flag patients who warrant a confirmatory diagnostic test. Machine learning is well suited to this kind of screening task because it can learn non-linear relationships between multiple risk factors simultaneously, rather than relying on a single threshold (e.g. fasting glucose alone).

This project uses the **DiaBD dataset**, a Bangladesh-specific clinical dataset, which is valuable precisely because diabetes risk factors and population characteristics (body size, blood pressure norms, disease prevalence) can differ meaningfully between populations — a model trained on a Western dataset (e.g. Pima Indians Diabetes Dataset) may not transfer well to a Bangladeshi clinical population. We make no exaggerated medical claims: the models in this project are trained for a coursework comparative-analysis exercise, and their predictions are **not clinical diagnoses**.

## 2. Objective

**Main objective:** To compare the performance of Logistic Regression, KNN, Random Forest, and XGBoost for diabetes classification using the DiaBD dataset.

**Secondary objectives:**
- Investigate the effect of severe class imbalance (~94% vs ~6%) on model performance.
- Compare two imbalance-handling strategies: class weighting and SMOTE.
- Identify which clinical/demographic features are most influential in each model's predictions.
- Compare the behavior of traditional statistical (Logistic Regression, KNN) and ensemble tree-based (Random Forest, XGBoost) algorithms.

## 3. Dataset Description

**Source file:** `data/diabd.csv` (copied from the downloaded `DiaBD A Diabetes Dataset for Enhanced Risk Analysi/` folder).

- **Instances (raw):** 5,288
- **Instances (after cleaning):** 5,276 (12 rows removed; see Section 4 and 9)
- **Features:** 14 predictor columns + 1 target column
- **Target:** `diabetic` (Yes/No), mapped to 1/0 for modelling
- **Missing values:** none found in any column
- **Duplicate rows:** none found

### Feature description table

| Feature | Type | Description | Observed range (post-cleaning) |
|---|---|---|---|
| age | Numerical | Patient age (years) | 21 – 80 |
| gender | Categorical | Female / Male | Female: 3,742 (70.9%), Male: 1,534 (29.1%) |
| pulse_rate | Numerical | Resting pulse rate (bpm) | 36 – 133 (after removing 3 impossible <30 bpm values) |
| systolic_bp | Numerical | Systolic blood pressure (mmHg) | 62 – 231 |
| diastolic_bp | Numerical | Diastolic blood pressure (mmHg) | 45 – 119 |
| glucose | Numerical | Blood glucose (mmol/L) | 0.04 – 33.46 (after removing 1 impossible 0 value) |
| height | Numerical | Height (m) | 1.22 – 1.96 (after widening the impossible-height threshold to <1.2 m, removing 6 rows) |
| weight | Numerical | Weight (kg) | 23.2 – 100.7 (after removing 2 impossible low-BMI rows: a 3 kg and a 21.9 kg adult) |
| bmi | Numerical | Body Mass Index | 11.47 – 54.08 (after removing 4 rows with bmi<10 or bmi>70; raw data ranged 1.22–574.13 due to the impossible height/weight rows removed above) |
| family_diabetes | Binary flag (0/1) | Family history of diabetes | 3.2% positive |
| hypertensive | Binary flag (0/1) | Currently hypertensive | 11.1% positive |
| family_hypertension | Binary flag (0/1) | Family history of hypertension | — |
| cardiovascular_disease | Binary flag (0/1) | History of cardiovascular disease | — |
| stroke | Binary flag (0/1) | History of stroke | — |
| **diabetic** (target) | Binary (Yes/No) | Diabetes status | No: 4,946 (93.53%), Yes: 342 (6.47%) — raw data |

### Class distribution

![Class Distribution](../outputs/figures/class_distribution.png)

The dataset is severely imbalanced: only 6.47% of patients (342 of 5,288) are labeled diabetic. This imbalance is the central methodological challenge addressed throughout this project.

## 4. Methodology

```text
DiaBD Dataset
     |
Data Inspection  (shape, dtypes, missing values, duplicates, describe)
     |
Data Cleaning    (remove 12 physiologically impossible rows; documented, not silent)
     |
Exploratory Data Analysis  (distributions, class-wise comparisons, correlation, outliers)
     |
Train/Test Split  (80/20, stratified on target, random_state=42)
     |
Encoding + Scaling  (OneHotEncoder for gender; StandardScaler for LR/KNN only; fit on
                      training data only, inside sklearn Pipelines, to prevent leakage)
     |
Baseline Models  (LR, KNN, RF, XGBoost trained on the untouched imbalanced training set)
     |
Class Imbalance Handling  (class_weight="balanced" vs SMOTE, compared against baseline)
     |
Hyperparameter Tuning  (5-fold StratifiedKFold CV, GridSearchCV/RandomizedSearchCV, scoring=F1)
     |
Final Evaluation  (tuned models scored once on the untouched test set)
     |
Feature Importance  (LR coefficients, RF/XGBoost importances)
     |
Model Comparison  (trade-off discussion; best model selected on evidence, not accuracy alone)
```

Each stage is implemented as a small, testable function in `src/` (`data_preprocessing.py`, `eda.py`, `train_models.py`, `evaluate_models.py`) and orchestrated end-to-end by `main.py`.

## 5. Machine Learning Algorithms

### Logistic Regression
Models the probability that a patient is diabetic as a sigmoid (logistic) function of a weighted sum of the input features: `P(y=1|x) = 1 / (1 + e^-(w·x + b))`. The weights `w` are learned by minimizing a regularized log-loss on the training data. It is a simple, fast, and interpretable linear baseline — its coefficients directly indicate how strongly (and in which direction) each feature is associated with the predicted log-odds of diabetes.

### K-Nearest Neighbors (KNN)
A non-parametric, instance-based method. To classify a new patient, KNN finds the `k` most similar patients in the training set (by distance in feature space) and predicts the majority class among them. It makes no assumption about the underlying data distribution, but is sensitive to feature scale (hence StandardScaler is required) and, as this project demonstrates, struggles when one class is rare, since a rare class is unlikely to dominate a random patient's `k` nearest neighbors.

### Random Forest
An ensemble of many decision trees, each trained on a bootstrap-resampled subset of the training data and a random subset of features at each split. Final predictions are made by majority vote (or averaged probability) across all trees. This reduces the overfitting risk of a single decision tree and can capture non-linear feature interactions.

### XGBoost (Extreme Gradient Boosting)
An ensemble of decision trees built **sequentially**, where each new tree is trained to correct the residual errors of the trees built so far (gradient boosting). This typically yields strong predictive performance and, via `scale_pos_weight`, offers a direct mechanism to up-weight the minority (diabetic) class during training.

## 6. Evaluation Metrics

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN) — overall proportion correct.
- **Precision** = TP / (TP + FP) — of patients predicted diabetic, how many actually are.
- **Recall** = TP / (TP + FN) — of patients who are actually diabetic, how many the model catches.
- **F1-score** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean, balances the two.
- **ROC-AUC** — the probability that the model ranks a randomly chosen diabetic patient's predicted risk higher than a randomly chosen non-diabetic patient's; threshold-independent.

**Why recall and F1 matter more than accuracy here:** with only 6.47% of patients diabetic, a trivial model that predicts "non-diabetic" for every patient scores ~93.5% accuracy while catching zero true diabetic cases. In a health-screening context, a **false negative** (telling a diabetic patient they are healthy) delays diagnosis and treatment, which is generally more costly than a **false positive** (referring a healthy patient for a confirmatory test). This is why diabetic-class recall and F1-score are emphasized throughout this project rather than accuracy alone.

## 7. Exploratory Data Analysis

All figures below are generated directly by `src/eda.py` and saved in `outputs/figures/`.

**Glucose** (`glucose_distribution.png`): Median glucose is 6.85 mmol/L for non-diabetic patients vs. 9.28 mmol/L for diabetic patients — a clear, medically expected separation, confirming glucose is a meaningful and correctly-scaled predictor (units are mmol/L, not mg/dL).

**BMI** (`bmi_distribution.png`): Median BMI is 21.79 for non-diabetic vs. 23.19 for diabetic patients — diabetic patients trend slightly higher, consistent with the known association between higher BMI and type-2 diabetes risk, though the separation is much less pronounced than for glucose.

**Age** (`age_distribution.png`): Median age is 45 for non-diabetic vs. 50 for diabetic patients, consistent with diabetes risk increasing with age.

**Hypertension** (`hypertensive_vs_target.png`): This is the single strongest categorical association found in the data. Among hypertensive patients (585 of 5,276, 11.1%), 31.3% are diabetic; among non-hypertensive patients, only 3.4% are diabetic. This large gap explains why `hypertensive` emerges as the top-ranked (or near-top-ranked) predictor in every model (Section 11).

**Gender** (`gender_vs_target.png`): Diabetes prevalence is broadly similar between genders in this sample (6.1% of female patients, 7.4% of male patients), a much smaller effect than hypertension status.

**Family history of diabetes** (`family_diabetes_vs_target.png`): Counter-intuitively, the `family_diabetes=1` subgroup (169 patients) showed a *lower* diabetes rate (3.55%, 6 of 169) than the `family_diabetes=0` subgroup (6.56%, 335 of 5,107). Given the very small size of the `family_diabetes=1` subgroup, this is likely a sampling artifact rather than a reliable population-level relationship, and should not be over-interpreted.

**Correlation heatmap** (`correlation_heatmap.png`, numerical features only — encoded flags and identifiers excluded): The strongest correlations are `weight`–`bmi` (r = 0.84, expected since BMI is computed from weight and height) and `systolic_bp`–`diastolic_bp` (r = 0.72, expected physiologically). `glucose` shows only weak linear correlation with other numerical features (|r| < 0.16 with all others), suggesting it contributes largely independent predictive signal — consistent with its strong showing in feature importance.

**Outlier analysis** (`outlier_boxplots.png`): Boxplots of all numerical features show a number of statistical outliers (e.g. some patients with very high systolic BP or glucose). These extreme-but-plausible values (e.g. glucose up to ~30 mmol/L, which is very high but does occur in poorly-controlled diabetics) were **retained** — they are medically plausible and informative for the classification task, unlike the physiologically impossible values described in Section 9, which were removed.

## 8. Class Imbalance

- **Raw class distribution:** No = 4,946 (93.53%), Yes = 342 (6.47%).
- **After stratified 80/20 split:** training set 93.53% / 6.47%, test set 93.56% / 6.44% — near-identical to the original ratio, confirming stratification worked correctly.

**Why accuracy is misleading here:** A classifier that always predicts "non-diabetic" would score 93.5% accuracy while achieving 0% recall on the class we actually care about detecting. This is close to the failure mode observed in the baseline KNN model, which scored 94.03% accuracy but only 14.71% diabetic-class recall (Section 10).

**Balancing approaches used:**
1. **Class weighting** — `class_weight="balanced"` for Logistic Regression and Random Forest (reweights the loss function to penalize misclassifying the minority class more heavily); `scale_pos_weight` (computed as `n_negative / n_positive` = 14.458 from the training data) for XGBoost. KNN has no equivalent parameter in scikit-learn, so it was left unweighted for this comparison (per the assignment's explicit instruction not to force an invalid approach onto KNN).
2. **SMOTE (Synthetic Minority Over-sampling Technique)** — generates synthetic minority-class (diabetic) samples by interpolating between existing minority-class neighbors. Applied **only to the training fold**, inside an `imblearn.pipeline.Pipeline`, so the test set and cross-validation validation folds are never touched by synthetic data (verified in Section 22).

## 9. Experimental Setup

- **Train/test split:** 80% / 20%, `random_state=42`, `stratify=y`.
- **Cleaning:** 12 rows removed for physiologically impossible values, identified via five conditions: glucose = 0 mmol/L (1 row, incompatible with a living patient); height < 1.2 m for an adult (6 rows: 0.36 m, 0.64 m, 0.99 m, and three separate rows sharing an identical, implausible 1.19 m / 55-60 kg pattern — two of the more extreme rows also produced BMI values of 574 and 156, confirming `bmi = weight / height²` in this dataset); pulse_rate < 30 bpm (3 rows: values of 5, 10, and 5 bpm, incompatible with life); and bmi < 10 or bmi > 70 (4 rows total, 2 of which overlap the height rule above — the two new catches were a 3 kg adult, bmi 1.22, and a 21.9 kg adult, bmi 8.29, neither of which was caught by the simpler height/glucose/pulse checks alone). The BMI-based check was added specifically because a raw `describe()` of the dataset shows a minimum adult weight of 3 kg, which is impossible but not flagged by height, glucose, or pulse_rate on its own — checking BMI catches implausible weight/height combinations directly. No duplicate rows were found. This is a targeted, documented removal of clear data-entry errors — not a blanket outlier-removal step (see Section 7 for outliers that were deliberately retained).
- **Encoding:** `gender` one-hot encoded (`drop="if_binary"`); the five already-binary flag columns (`family_diabetes`, `hypertensive`, `family_hypertension`, `cardiovascular_disease`, `stroke`) passed through unchanged.
- **Scaling:** `StandardScaler` for Logistic Regression and KNN (fit on the training fold only, inside each model's Pipeline); Random Forest and XGBoost use unscaled features, since tree-based splits are invariant to monotonic feature scaling.
- **Cross-validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for all hyperparameter tuning.
- **Hyperparameter tuning:** `GridSearchCV` (exhaustive) for Logistic Regression and KNN, whose search spaces are small; `RandomizedSearchCV` (`n_iter=30`) for Random Forest and XGBoost, whose search spaces are larger. Scoring metric: **F1-score** on the diabetic class (justification in Section 6).
- **SMOTE:** applied only inside training folds via `imblearn.pipeline.Pipeline`, `random_state=42`.

## 10. Results

### 10.1 Baseline Model Results (no imbalance handling)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9384 | 0.5600 | 0.2059 | 0.3011 | 0.8448 |
| KNN | 0.9403 | 0.6667 | 0.1471 | 0.2410 | 0.7031 |
| Random Forest | 0.9384 | 0.5789 | 0.1618 | 0.2529 | 0.8694 |
| XGBoost | 0.9394 | 0.5526 | 0.3088 | 0.3962 | 0.8761 |

*(`outputs/tables/baseline_results.csv`; confusion matrices: `outputs/figures/confusion_matrices_baseline.png`; ROC curves: `outputs/figures/roc_curves_baseline.png`)*

All four baseline models score within 0.2 points of each other on accuracy (93.84–94.03%), yet their diabetic-class recall ranges from a modest 14.7% (KNN) to 30.9% (XGBoost). This is the clearest demonstration in this project of why accuracy alone is an unreliable metric for imbalanced classification.

### 10.2 Balanced / SMOTE Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.9384 | 0.5600 | 0.2059 | 0.3011 | 0.8448 |
| Logistic Regression (Class-Weighted) | 0.8305 | 0.2239 | **0.6618** | 0.3346 | 0.8426 |
| Logistic Regression (SMOTE) | 0.8333 | 0.2216 | 0.6324 | 0.3282 | 0.8396 |
| KNN (Baseline) | 0.9403 | 0.6667 | 0.1471 | 0.2410 | 0.7031 |
| KNN (Class-Weighted, n/a) | 0.9403 | 0.6667 | 0.1471 | 0.2410 | 0.7031 |
| KNN (SMOTE) | 0.8277 | 0.1968 | 0.5441 | 0.2891 | 0.7682 |
| Random Forest (Baseline) | 0.9384 | 0.5789 | 0.1618 | 0.2529 | 0.8694 |
| Random Forest (Class-Weighted) | 0.9337 | 0.4444 | 0.1176 | 0.1860 | 0.8662 |
| Random Forest (SMOTE) | 0.9271 | 0.4182 | 0.3382 | 0.3740 | 0.8445 |
| XGBoost (Baseline) | 0.9394 | 0.5526 | 0.3088 | 0.3962 | **0.8761** |
| XGBoost (Class-Weighted) | 0.9347 | 0.4915 | 0.4265 | **0.4567** | 0.8642 |
| XGBoost (SMOTE) | 0.9375 | **0.5208** | 0.3676 | 0.4310 | 0.8513 |

*(`outputs/tables/balanced_results.csv`)*

Key observations:
- **Class weighting had a dramatic effect on Logistic Regression** (recall 20.6% → 66.2%, F1 0.301 → 0.335) but actually *reduced* Random Forest's recall (16.2% → 11.8%, F1 0.253 → 0.186) — plain `class_weight="balanced"` alone was not just insufficient for Random Forest, it made its test-set performance worse; the tuned Random Forest in Section 10.3 (which searches `max_depth`, `min_samples_leaf`, etc. jointly alongside `class_weight`) is what actually delivers a recall improvement.
- **SMOTE improved recall for every single model**, most dramatically for KNN (14.7% → 54.4%), consistent with a meaningful part of KNN's weak baseline recall being an imbalance problem (with more synthetic minority neighbors available, KNN can now find diabetic neighbors).
- **Class weighting also had a striking effect on XGBoost**, raising its F1 from 0.396 to 0.457 (the highest F1 in this whole table, tuned models included) by lifting recall from 30.9% to 42.6% while giving up less precision than the other imbalance strategies.
- **Most imbalance-handling strategies traded accuracy for recall**, as expected on a dataset where high accuracy is easy but uninformative. Random Forest's class-weighted result is the exception that proves the rule: it lost on both accuracy (93.8% → 93.4%) *and* recall (16.2% → 11.8%) at once, showing that `class_weight="balanced"` is not automatically beneficial in isolation.

### 10.3 Hyperparameter Tuning (5-fold CV, scoring = F1)

| Model | Best CV F1 | Best Parameters |
|---|---|---|
| Logistic Regression | 0.3726 | `C=10`, `penalty='l1'`, `class_weight='balanced'` |
| KNN | 0.2806 | `n_neighbors=3`, `weights='distance'`, `p=2` |
| Random Forest | 0.4388 | `n_estimators=100`, `max_depth=30`, `min_samples_split=5`, `min_samples_leaf=4`, `max_features='log2'`, `class_weight='balanced'` |
| XGBoost | 0.4148 | `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.7`, `colsample_bytree=0.7`, `scale_pos_weight=1` |

*(`outputs/tables/best_hyperparameters.csv`)*

Notably, cross-validation **independently selected `class_weight="balanced"`** for both Logistic Regression and Random Forest — the search was free to keep these two models unweighted if that scored better on F1, and it did not, which is evidence, not an assumption, that imbalance-aware weighting helps them on this dataset. XGBoost is the interesting exception: the search instead selected `scale_pos_weight=1` (i.e. no weighting at all), relying purely on `subsample`/`colsample_bytree`/a lower `learning_rate` to control overfitting to the majority class. This does not mean weighting never helps XGBoost — Section 10.2 shows a hand-picked `class_weight`-equivalent (`scale_pos_weight=14.458`) achieved the single highest F1 (0.4567) of any configuration in this entire study, tuned models included — it means the specific 30-iteration `RandomizedSearchCV` search used here did not land on that region of the joint hyperparameter space. This is flagged explicitly in Section 12 and Section 13 rather than glossed over.

### 10.4 Final Comparison (Tuned Models on the Untouched Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8305 | 0.2239 | **0.6618** | 0.3346 | 0.8427 |
| KNN | 0.9299 | 0.3929 | 0.1618 | 0.2292 | 0.6722 |
| Random Forest | 0.9271 | 0.4286 | 0.3971 | **0.4122** | 0.8675 |
| XGBoost | 0.9375 | **0.5312** | 0.2500 | 0.3400 | **0.8813** |

*(`outputs/tables/tuned_results.csv`, `outputs/tables/final_comparison.csv`; confusion matrices: `outputs/figures/confusion_matrices_tuned.png`; ROC curves: `outputs/figures/roc_curves.png`; bar chart: `outputs/figures/model_comparison.png`)*

**Ranking by different criteria:**
- **Highest recall (best at catching diabetic patients):** Logistic Regression (66.2%) — by a wide margin.
- **Highest precision (fewest false alarms among positive predictions):** XGBoost (53.1%), but at only 25.0% recall — a large share of diabetic patients would still be missed if precision alone were prioritized.
- **Highest F1-score (best precision/recall balance):** Random Forest (0.4122).
- **Highest ROC-AUC (best overall ranking ability, threshold-independent):** XGBoost (0.8813), Random Forest close behind (0.8675).
- **Simplest / fastest / most interpretable:** Logistic Regression (linear coefficients, fastest to train and predict).

No single model dominates on every metric — this is discussed further in Section 12.

## 11. Feature Importance

*(Association with model predictions — not a claim of medical causation. All values are read directly from `outputs/tables/feature_importance_*.csv`.)*

**Logistic Regression** (standardized |coefficient|, top 5): `hypertensive` (2.26), `family_hypertension` (2.11), `family_diabetes` (2.08, negative direction — see note below), `bmi` (1.25), `cardiovascular_disease` (0.86). `glucose` (0.71) ranks 8th.

**Random Forest** (feature importance, top 5): `glucose` (0.231), `hypertensive` (0.175), `systolic_bp` (0.113), `diastolic_bp` (0.093), `weight` (0.090).

**XGBoost** (feature importance, top 5): `hypertensive` (0.474), `glucose` (0.103), `diastolic_bp` (0.046), `age` (0.045), `systolic_bp` (0.044).

![Feature Importance (XGBoost)](../outputs/figures/feature_importance.png)

**Discussion:** `hypertensive` status is the single most consistently influential feature — ranked #1 for Logistic Regression and XGBoost, and #2 for Random Forest — directly consistent with the EDA finding in Section 7 that diabetes prevalence is roughly 9× higher among hypertensive patients (31.3%) than non-hypertensive patients (3.4%) in this dataset, a strong and medically plausible signal (hypertension and type-2 diabetes are well-documented comorbid conditions). `glucose` is the top feature for Random Forest and the #2 feature for XGBoost, but only the 8th-ranked feature for Logistic Regression — most likely because its linear signal is split across several numerical features that correlate with it once standardized (`bmi`, `weight`, and `height` all outrank it for Logistic Regression), an effect tree-based models are less prone to since each split simply picks whichever single feature best separates the classes at that node, regardless of what else correlates with it. Note also that Logistic Regression's `family_diabetes` coefficient is *negative* (family history associated with **lower** predicted risk in this specific dataset) — this mirrors the small-subgroup anomaly already flagged in Section 7 (`family_diabetes=1`, n=169) and should be read as a data-driven artifact of a small subgroup, not a medically meaningful reversal. All three models' importances are read as **associated with**, not necessarily causing, diabetes status.

## 12. Discussion

**Which model performed best?** There is no single "best" model independent of use case:
- If the priority is **not missing diabetic patients** (a first-pass screening tool, where a false positive only costs a follow-up test), **Logistic Regression** is the strongest choice: it catches 66.2% of diabetic patients in the test set, more than 1.6× any other tuned model, and remains the simplest and most interpretable model.
- If the priority is a **balanced trade-off** between catching diabetic patients and not overwhelming clinicians with false alarms, **Random Forest** is the strongest choice: it has the best tuned F1-score (0.4122) and a strong ROC-AUC (0.8675).
- If the priority is **overall ranking quality** (e.g. for triaging patients by risk score rather than a hard yes/no cutoff), **XGBoost** is the strongest choice, with the best ROC-AUC (0.8813).
- **KNN was consistently the weakest model** on every imbalance-sensitive metric (baseline recall 14.7%, tuned recall 16.2%, tuned F1 0.2292, tuned ROC-AUC 0.6722 — all the lowest, or tied lowest, of any model in the final tuned comparison) — likely because it has no built-in mechanism to account for class imbalance; its SMOTE variant, while much better (54.4% recall), still trailed the tree-based models on F1 and precision.

**Did balancing improve minority-class prediction?** Mostly yes. Class weighting and/or SMOTE improved diabetic-class recall for Logistic Regression, KNN, and XGBoost relative to their unweighted baselines (Section 10.2). The clear exception was Random Forest under plain class weighting alone, whose recall actually *fell* (16.2% → 11.8%); Random Forest only benefited from imbalance handling once tuned jointly with its other hyperparameters (Section 10.3–10.4).

**Did tuning improve results?** Mixed, and this project's tuning results carry a genuine lesson about the gap between cross-validation and held-out test performance. Tuning clearly improved Random Forest's test F1 (0.2529 → 0.4122) and recall (16.2% → 39.7%). But **XGBoost's test F1 actually got *worse* after tuning (0.3962 → 0.3400)**, even though the search's chosen configuration had the best cross-validated F1 (0.4148) among the non-Random-Forest candidates — those hyperparameters generalized less well to the held-out test set than XGBoost's untuned defaults did. **KNN's tuned ROC-AUC (0.6722) was also worse than its untuned baseline (0.7031)**, for a related reason: cross-validation selected `n_neighbors=3` to maximize F1-score on the training folds, a high-variance choice that did not transfer as cleanly to ROC-AUC on the held-out test set. Together, these two results are a useful illustration that optimizing a metric via cross-validation does not guarantee improvement on that same metric — or on a different metric — on genuinely unseen data, and that the choice of tuning metric and search space both matter (Section 6, Section 10.3).

**Possible signs of overfitting:** Random Forest's tuned configuration (`max_depth=30`, `min_samples_leaf=4`) allows fairly deep, expressive trees, yet its test-set ROC-AUC (0.8675) stayed close to its baseline (0.8694), suggesting this did not meaningfully harm its actual generalization on this dataset. XGBoost's test-F1 regression above is the clearer example of a metric-driven overfitting effect in this project: its cross-validated F1 (0.4148) overstated how well the chosen hyperparameters would generalize. Relatedly, Section 10.2 showed that a simple, hand-picked class-weighted XGBoost (no other tuning) reached F1 = 0.4567 — higher than *any* model in the final tuned comparison — which suggests the `RandomizedSearchCV(n_iter=30)` search for XGBoost did not fully explore the region of hyperparameter space where weighting helps most; a larger `n_iter` or a search space that couples `scale_pos_weight` more tightly with regularization parameters might close this gap.

**Do simpler models perform competitively?** Partially. Logistic Regression, the simplest model in this comparison, achieved by far the highest recall of any tuned model, but its ROC-AUC (0.8427) trailed both tree-based ensembles (0.8675–0.8813) by a more noticeable margin than a purely-linear relationship would predict, suggesting the relationship between these clinical features and diabetes status is not purely linear.

## 13. Limitations

- **Severe class imbalance** (6.47% diabetic) means even the best recall achieved (66.2%, Logistic Regression) still misses roughly a third of diabetic patients, and all models' precision remains low (22–53%), meaning a meaningful fraction of positive predictions are false alarms.
- **Single-country, single-dataset sample:** the DiaBD dataset reflects one Bangladeshi clinical population; findings (e.g. the strength of the hypertension–diabetes association) may not generalize to other populations without external validation.
- **Predictions are not clinical diagnoses.** These models are trained for a coursework comparative-analysis exercise and should not be used, as-is, for real clinical decision-making.
- **Limited feature set:** the dataset does not include some risk factors known to be relevant to diabetes (e.g. diet, physical activity level, HbA1c, waist circumference), which likely constrains achievable performance.
- **A small subgroup anomaly** was observed for `family_diabetes` (Section 7) — the `family_diabetes=1` subgroup is small (169 patients) and showed a lower-than-expected diabetes rate, which is more likely a sampling artifact than a reliable finding, and should not be treated as evidence that family history is protective.
- **Tuned results can regress relative to cross-validation** (Section 12) — XGBoost's test F1 fell after tuning despite a strong CV score, and KNN's test ROC-AUC fell similarly; both are concrete reminders that CV performance is an estimate of generalization, not a guarantee, especially with a minority class this small (~340 positive cases split across 5 folds).
- **Model generalizability requires external validation** on an independent dataset before any of these findings should be treated as robust beyond this specific sample and train/test split.

## 14. Conclusion

This project compared Logistic Regression, KNN, Random Forest, and XGBoost for diabetes risk classification on the DiaBD Bangladesh dataset, with a specific focus on the dataset's severe class imbalance (6.47% diabetic). Baseline models achieved misleadingly high accuracy (93.8–94.0%) while catching relatively few diabetic patients (14.7–30.9% recall), directly demonstrating why accuracy alone is an inappropriate metric for this task. Class weighting and SMOTE meaningfully improved diabetic-class recall for most models, though Random Forest needed full hyperparameter tuning — not class weighting alone — to actually benefit; cross-validated tuning independently confirmed that imbalance-aware weighting improves Logistic Regression and Random Forest on this dataset. After tuning, no single algorithm dominated every metric: **Random Forest offered the best precision/recall balance (F1 = 0.412)**, **XGBoost offered the best overall ranking ability (ROC-AUC = 0.881)**, and **Logistic Regression offered by far the highest diabetic-class recall (66.2%)** at the cost of precision. `hypertensive` status was the single most consistently important predictor across all interpretable models, and `glucose` was a close second specifically for the two tree-based models — findings directly grounded in, and consistent with, the underlying data. Tuning did not universally help: both XGBoost's and KNN's test performance regressed on at least one metric relative to their untuned baselines, despite good cross-validated scores — a useful reminder that cross-validated hyperparameter search optimizes an *estimate* of generalization, not generalization itself. The main research question — how do these four algorithms compare, and how does imbalance handling affect that comparison — is answered with clear, reproducible, data-grounded evidence: **model selection for this task should be driven by the intended screening use case (favoring recall vs. favoring precision/balance/ranking quality) rather than by accuracy or by picking a single "winner" algorithm.**

## 15. Future Work

- Collect a larger and/or more geographically diverse Bangladesh diabetes dataset to increase the number of positive (diabetic) cases available for training and to test generalizability across regions.
- Perform external validation on an independent dataset (e.g. a different hospital or region) before considering any deployment.
- Incorporate additional lifestyle and clinical variables (diet, physical activity, HbA1c, waist circumference) that are known diabetes risk factors but are absent from this dataset.
- Apply explainable AI techniques (e.g. SHAP values) for a more rigorous, per-patient interpretability analysis beyond global feature importance.
- If deployed as a research prototype, build a simple risk-score interface around the saved Random Forest/XGBoost pipeline (`outputs/models/`), clearly labeled as a research tool and not a diagnostic device.
- Compare against deep learning approaches (e.g. a small feed-forward neural network) if a substantially larger dataset becomes available — with only 342 positive cases in the current dataset, deep learning is not justified here and would likely overfit.

---

*All figures referenced above are saved in `outputs/figures/`; all tables are saved in `outputs/tables/`; all numbers in this report were generated by running `python main.py` on the cleaned DiaBD dataset (`random_state=42` throughout) and are reproducible by re-running that command.*
