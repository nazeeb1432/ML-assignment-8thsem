"""
main.py — End-to-end pipeline for the DiaBD diabetes classification project.

Run with:
    python main.py

This script performs the complete workflow described in report/final_report.md:
    1. Load and inspect the raw dataset
    2. Clean the data (documented removal of impossible values only)
    3. Exploratory Data Analysis (figures saved to outputs/figures/)
    4. Train/test split (80/20, stratified, random_state=42)
    5. Baseline models (Logistic Regression, KNN, Random Forest, XGBoost)
    6. Class imbalance handling (class weighting vs SMOTE)
    7. Hyperparameter tuning (GridSearchCV / RandomizedSearchCV, scoring=F1)
    8. Final evaluation of tuned models on the untouched test set
    9. Feature importance analysis
    10. Saving the best model + all tables/figures to outputs/
"""

import time

import joblib
import pandas as pd

from src import eda, evaluate_models as ev, utils
from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    encode_target,
    load_data,
    split_data,
)
from src.train_models import (
    build_pipelines,
    build_smote_pipelines,
    compute_scale_pos_weight,
    fit_all,
    tune_all,
)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    start_time = time.time()
    utils.ensure_output_dirs()

    # ------------------------------------------------------------------
    # 1. Load and inspect the raw dataset
    # ------------------------------------------------------------------
    section("STEP 1: LOAD AND INSPECT DATASET")
    df_raw = load_data()
    print(f"Dataset file: {utils.DATA_PATH}")
    print(f"Shape: {df_raw.shape}")
    print("\nColumns and dtypes:")
    print(df_raw.dtypes)
    print("\nHead:")
    print(df_raw.head())
    print("\nMissing values per column:")
    print(df_raw.isnull().sum())
    print(f"\nDuplicate rows: {df_raw.duplicated().sum()}")
    print("\nTarget ('diabetic') value counts:")
    print(df_raw["diabetic"].value_counts())
    print("\nTarget ('diabetic') percentages:")
    print((df_raw["diabetic"].value_counts(normalize=True) * 100).round(2))
    print("\nDescribe (numerical features):")
    print(df_raw.describe().round(2))

    # ------------------------------------------------------------------
    # 2. Clean the data
    # ------------------------------------------------------------------
    section("STEP 2: DATA CLEANING")
    df_clean = clean_data(df_raw, verbose=True)

    # ------------------------------------------------------------------
    # 3. Exploratory Data Analysis
    # ------------------------------------------------------------------
    section("STEP 3: EXPLORATORY DATA ANALYSIS")
    # Use a copy with a readable target column name for plotting.
    df_eda = df_clean.rename(columns={"diabetic": "diabetic_label"})

    fig_paths = []
    fig_paths.append(eda.plot_class_distribution(df_eda))
    for col in ["glucose", "bmi", "age", "systolic_bp", "diastolic_bp", "pulse_rate", "weight", "height"]:
        fig_paths.append(eda.plot_numeric_distribution(df_eda, col))
    for col in ["gender", "family_diabetes", "hypertensive", "family_hypertension", "cardiovascular_disease", "stroke"]:
        fig_paths.append(eda.plot_categorical_vs_target(df_eda, col))
    fig_paths.append(eda.plot_correlation_heatmap(df_clean))
    fig_paths.append(eda.plot_boxplots_outliers(df_clean))
    print(f"Saved {len(fig_paths)} EDA figures to {utils.FIGURES_DIR}")

    # ------------------------------------------------------------------
    # 4. Encode target and split
    # ------------------------------------------------------------------
    section("STEP 4: TARGET ENCODING AND TRAIN/TEST SPLIT")
    df_model = encode_target(df_clean)
    X_train, X_test, y_train, y_test = split_data(df_model)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print("Train class distribution (%):")
    print((y_train.value_counts(normalize=True) * 100).round(2))
    print("Test class distribution (%):")
    print((y_test.value_counts(normalize=True) * 100).round(2))

    # ------------------------------------------------------------------
    # 5. Baseline models
    # ------------------------------------------------------------------
    section("STEP 5: BASELINE MODELS (NO IMBALANCE HANDLING)")
    baseline_pipelines = build_pipelines()
    baseline_fitted = fit_all(baseline_pipelines, X_train, y_train)
    baseline_results = ev.build_comparison_table(baseline_fitted, X_test, y_test)
    print(baseline_results.round(4))
    ev.print_classification_reports(baseline_fitted, X_test, y_test)

    baseline_results.round(4).to_csv(f"{utils.TABLES_DIR}/baseline_results.csv")
    ev.plot_confusion_matrices(baseline_fitted, X_test, y_test, filename="confusion_matrices_baseline.png")
    ev.plot_roc_curves(baseline_fitted, X_test, y_test, filename="roc_curves_baseline.png")
    ev.plot_model_comparison_bar(baseline_results, filename="model_comparison_baseline.png")

    # ------------------------------------------------------------------
    # 6. Class imbalance handling: class weighting vs SMOTE
    # ------------------------------------------------------------------
    section("STEP 6: CLASS IMBALANCE HANDLING")
    scale_pos_weight_value = compute_scale_pos_weight(y_train)
    print(f"Computed XGBoost scale_pos_weight from training data: {scale_pos_weight_value:.3f}")

    weighted_pipelines = build_pipelines(class_weight="balanced", scale_pos_weight=scale_pos_weight_value)
    weighted_fitted = fit_all(weighted_pipelines, X_train, y_train)
    weighted_results = ev.build_comparison_table(weighted_fitted, X_test, y_test)
    weighted_results.index = [f"{name} (Class-Weighted)" for name in weighted_results.index]

    smote_pipelines = build_smote_pipelines()
    smote_fitted = fit_all(smote_pipelines, X_train, y_train)
    smote_results = ev.build_comparison_table(smote_fitted, X_test, y_test)
    smote_results.index = [f"{name} (SMOTE)" for name in smote_results.index]

    baseline_labeled = baseline_results.copy()
    baseline_labeled.index = [f"{name} (Baseline)" for name in baseline_labeled.index]

    balanced_results = pd.concat([baseline_labeled, weighted_results, smote_results])
    print(balanced_results.round(4))
    balanced_results.round(4).to_csv(f"{utils.TABLES_DIR}/balanced_results.csv")

    # ------------------------------------------------------------------
    # 7. Hyperparameter tuning
    # ------------------------------------------------------------------
    section("STEP 7: HYPERPARAMETER TUNING (5-fold CV, scoring=F1)")
    searches = tune_all(X_train, y_train, n_iter=30)

    best_params_rows = []
    tuned_fitted = {}
    for name, search in searches.items():
        print(f"\n{name}:")
        print(f"  Best CV F1 score: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")
        tuned_fitted[name] = search.best_estimator_
        best_params_rows.append(
            {"Model": name, "Best_CV_F1": search.best_score_, "Best_Params": str(search.best_params_)}
        )

    best_params_df = pd.DataFrame(best_params_rows).set_index("Model")
    best_params_df.to_csv(f"{utils.TABLES_DIR}/best_hyperparameters.csv")

    # ------------------------------------------------------------------
    # 8. Final evaluation of tuned models on the untouched test set
    # ------------------------------------------------------------------
    section("STEP 8: FINAL EVALUATION (TUNED MODELS ON TEST SET)")
    tuned_results = ev.build_comparison_table(tuned_fitted, X_test, y_test)
    tuned_results = tuned_results.join(best_params_df[["Best_Params"]])
    print(tuned_results.round(4))
    ev.print_classification_reports(tuned_fitted, X_test, y_test)

    tuned_results.to_csv(f"{utils.TABLES_DIR}/tuned_results.csv")
    ev.plot_confusion_matrices(tuned_fitted, X_test, y_test, filename="confusion_matrices_tuned.png")
    ev.plot_roc_curves(tuned_fitted, X_test, y_test, filename="roc_curves.png")
    ev.plot_model_comparison_bar(tuned_results, filename="model_comparison.png")

    # Final comparison table = baseline vs tuned, side by side, for the report.
    final_comparison = pd.concat(
        [
            baseline_results.round(4).add_suffix("_Baseline"),
            tuned_results[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].round(4).add_suffix("_Tuned"),
        ],
        axis=1,
    )
    final_comparison.to_csv(f"{utils.TABLES_DIR}/final_comparison.csv")
    print("\nFinal Comparison (Baseline vs Tuned):")
    print(final_comparison)

    # ------------------------------------------------------------------
    # 9. Feature importance
    # ------------------------------------------------------------------
    section("STEP 9: FEATURE IMPORTANCE")
    lr_importance = ev.get_logistic_regression_importance(tuned_fitted["Logistic Regression"])
    rf_importance = ev.get_tree_importance(tuned_fitted["Random Forest"])
    xgb_importance = ev.get_tree_importance(tuned_fitted["XGBoost"])

    print("\nTop Logistic Regression coefficients (|value|):")
    print(lr_importance.head(10))
    print("\nTop Random Forest importances:")
    print(rf_importance.head(10))
    print("\nTop XGBoost importances:")
    print(xgb_importance.head(10))

    lr_importance.to_csv(f"{utils.TABLES_DIR}/feature_importance_logreg.csv", index=False)
    rf_importance.to_csv(f"{utils.TABLES_DIR}/feature_importance_rf.csv", index=False)
    xgb_importance.to_csv(f"{utils.TABLES_DIR}/feature_importance_xgb.csv", index=False)

    ev.plot_feature_importance(
        lr_importance, "Abs_Coefficient", "Logistic Regression: Top Features (|Coefficient|)",
        "feature_importance_logreg.png",
    )
    ev.plot_feature_importance(
        rf_importance, "Importance", "Random Forest: Top Features", "feature_importance_rf.png",
    )
    ev.plot_feature_importance(
        xgb_importance, "Importance", "XGBoost: Top Features (Feature Importance)", "feature_importance.png",
    )

    # ------------------------------------------------------------------
    # 10. Save the best overall model
    # ------------------------------------------------------------------
    section("STEP 10: SAVE BEST MODEL")
    # Best model chosen by F1-score on the diabetic class (balances
    # precision and recall) among the tuned models. See report/final_report.md
    # Section 12 for the full trade-off discussion (recall, ROC-AUC, etc.).
    best_model_name = tuned_results["F1 Score"].idxmax()
    best_model = tuned_fitted[best_model_name]
    print(f"Best model by test F1-score: {best_model_name}")
    print(tuned_results.loc[best_model_name])

    joblib.dump(best_model, f"{utils.MODELS_DIR}/best_model.joblib")
    for name, pipeline in tuned_fitted.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(pipeline, f"{utils.MODELS_DIR}/{safe_name}_tuned.joblib")
    print(f"Saved best model ('{best_model_name}') and all tuned pipelines to {utils.MODELS_DIR}")

    elapsed = time.time() - start_time
    section(f"PIPELINE COMPLETE in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
