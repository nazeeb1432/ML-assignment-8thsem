"""
Model definitions and training helpers for the four algorithms compared
in this project: Logistic Regression, KNN, Random Forest, and XGBoost.

Every model is wrapped in an sklearn Pipeline together with its
preprocessing step, so that:
  - StandardScaler (for Logistic Regression / KNN) is fit only on the
    training fold, never on test data.
  - The exact same preprocessing is reused automatically during
    cross-validation and hyperparameter search.
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_preprocessing import build_preprocessor
from src.utils import RANDOM_STATE


def build_pipelines(class_weight=None, scale_pos_weight=None):
    """
    Build the four model pipelines.

    class_weight:
        None            -> baseline models (imbalance ignored)
        "balanced"      -> Logistic Regression and Random Forest reweight
                            classes inversely proportional to frequency
    scale_pos_weight:
        A float used only by XGBoost to up-weight the minority (diabetic)
        class. If None, XGBoost trains with its default weighting.

    KNN has no direct class-weighting mechanism in scikit-learn, so it is
    always left at its default settings here (per assignment instructions,
    we do not force an invalid approach onto KNN).

    Returns a dict: {model_name: sklearn Pipeline}
    """
    pipelines = {}

    # --- Logistic Regression: needs scaled features ---
    # solver="liblinear" is used because it is numerically stable on this
    # dataset; the lbfgs/newton-cg solvers raise harmless but noisy
    # RuntimeWarnings (divide-by-zero in intermediate steps) caused by the
    # dataset's severe class imbalance without changing the final result.
    pipelines["Logistic Regression"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    solver="liblinear",
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    # --- KNN: needs scaled features, no class_weight support ---
    pipelines["KNN"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    # --- Random Forest: tree-based, no scaling needed ---
    pipelines["Random Forest"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=200,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    # --- XGBoost: tree-based, no scaling needed ---
    xgb_kwargs = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    if scale_pos_weight is not None:
        xgb_kwargs["scale_pos_weight"] = scale_pos_weight

    pipelines["XGBoost"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("classifier", XGBClassifier(**xgb_kwargs)),
        ]
    )

    return pipelines


def build_smote_pipelines():
    """
    Build the four model pipelines with SMOTE inserted between
    preprocessing and the classifier, using imblearn's Pipeline so that
    SMOTE is applied only inside the training folds (never at predict
    time / never to the test set).
    """
    pipelines = {}

    pipelines["Logistic Regression"] = ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver="liblinear")),
        ]
    )

    pipelines["KNN"] = ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    pipelines["Random Forest"] = ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200)),
        ]
    )

    pipelines["XGBoost"] = ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return pipelines


def compute_scale_pos_weight(y_train) -> float:
    """
    scale_pos_weight for XGBoost is conventionally:
        (number of negative samples) / (number of positive samples)
    computed from the TRAINING labels only.
    """
    n_negative = int(np.sum(y_train == 0))
    n_positive = int(np.sum(y_train == 1))
    return n_negative / n_positive


def fit_pipeline(pipeline, X_train, y_train):
    """Fit a single pipeline and return it."""
    pipeline.fit(X_train, y_train)
    return pipeline


def fit_all(pipelines: dict, X_train, y_train) -> dict:
    """Fit every pipeline in the dict and return the fitted dict."""
    fitted = {}
    for name, pipeline in pipelines.items():
        fitted[name] = fit_pipeline(pipeline, X_train, y_train)
    return fitted


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
# We tune with 5-fold StratifiedKFold cross-validation, scored on F1 for the
# diabetic (positive) class. F1 is chosen over accuracy because accuracy is
# misleading on this ~94%/6% imbalanced dataset (a model that predicts
# "non-diabetic" for everyone still scores ~94% accuracy). F1 is also
# preferred over recall alone or ROC-AUC alone here because:
#   - optimising for recall alone can be "gamed" by a model that predicts
#     the positive class very liberally, tanking precision;
#   - ROC-AUC is threshold-independent and can look good even when the
#     model's actual (thresholded) predictions have a poor precision/recall
#     balance.
# F1 forces a balance between catching diabetic cases (recall) and not
# overwhelming clinicians with false alarms (precision).
TUNING_SCORING = "f1"
CV_SPLITS = 5


def _cv():
    """Stratified 5-fold CV splitter, reproducible via RANDOM_STATE."""
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def get_param_grids(scale_pos_weight_value: float):
    """
    Hyperparameter search spaces for each model.

    class_weight / scale_pos_weight are included as tunable options for
    Logistic Regression, Random Forest, and XGBoost, so cross-validation
    itself decides whether imbalance-aware weighting improves the F1 score
    - rather than us assuming it always helps. KNN has no such option
    (scikit-learn's KNeighborsClassifier does not support class weighting),
    so its grid only covers its own hyperparameters.
    """
    grids = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l1", "l2"],
            "classifier__class_weight": [None, "balanced"],
        },
        "KNN": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__class_weight": [None, "balanced"],
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.7, 0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 1.0],
            "classifier__scale_pos_weight": [1, scale_pos_weight_value],
        },
    }
    return grids


def tune_model(name, pipeline, param_grid, X_train, y_train, n_iter=25):
    """
    Run hyperparameter search for one model.

    Logistic Regression and KNN use GridSearchCV because their search
    spaces are small enough to search exhaustively. Random Forest and
    XGBoost use RandomizedSearchCV with a capped number of iterations
    (n_iter) so the search space stays large (realistic) while total
    runtime stays reasonable on a normal laptop.
    """
    if name in ("Logistic Regression", "KNN"):
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=TUNING_SCORING,
            cv=_cv(),
            n_jobs=-1,
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=TUNING_SCORING,
            cv=_cv(),
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    search.fit(X_train, y_train)
    return search


def tune_all(X_train, y_train, n_iter=25) -> dict:
    """
    Tune all four models and return a dict of fitted search objects, each
    exposing .best_estimator_, .best_params_, and .best_score_.
    """
    scale_pos_weight_value = compute_scale_pos_weight(y_train)
    grids = get_param_grids(scale_pos_weight_value)
    # class_weight="balanced" for Logistic Regression uses solver="liblinear"
    # which supports both l1 and l2 penalties, matching the tuned grid above.
    base_pipelines = build_pipelines()

    searches = {}
    for name, pipeline in base_pipelines.items():
        searches[name] = tune_model(name, pipeline, grids[name], X_train, y_train, n_iter=n_iter)
    return searches
