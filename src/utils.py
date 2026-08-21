"""
Shared constants and small helper functions used across the project.

Keeping these in one place means every script (data_preprocessing.py,
train_models.py, evaluate_models.py, main.py, and the notebook) agrees on
things like file paths and the random seed, instead of repeating magic
values everywhere.
"""

import os
import warnings

# On macOS, numpy is often built against the Accelerate BLAS backend, which
# is known to raise spurious "divide by zero / overflow / invalid value
# encountered in matmul" RuntimeWarnings during LogisticRegression's
# predict_proba (e.g. via scikit-learn's expit/softmax matrix multiply).
# This was verified on this project to be a harmless numerical artifact:
# the resulting probabilities contain no NaN/Inf and match expected ranges.
# We silence only these three specific matmul messages so real warnings
# (e.g. convergence issues) are still shown.
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

# imbalanced-learn 0.12.4's SMOTE predates scikit-learn 1.6's internal API
# changes (_validate_data -> validate_data, _get_tags -> __sklearn_tags__).
# This is a library version-compatibility warning, not a bug in this
# project's code, and does not affect SMOTE's behaviour or correctness.
warnings.filterwarnings("ignore", message=".*_validate_data.*is deprecated in 1.6.*")
warnings.filterwarnings("ignore", message=".*_get_tags.*and.*_more_tags.*")

# GridSearchCV/RandomizedSearchCV with n_jobs=-1 run folds in separate
# worker processes that do not inherit the filters registered above via
# warnings.filterwarnings(). Setting PYTHONWARNINGS propagates the same
# three benign-and-verified suppressions to those worker processes.
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore:divide by zero encountered in matmul:RuntimeWarning"
    ",ignore:overflow encountered in matmul:RuntimeWarning"
    ",ignore:invalid value encountered in matmul:RuntimeWarning",
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Using one fixed seed everywhere means the train/test split, model
# initialisation, SMOTE sampling, and cross-validation folds are identical
# every time the code is run.
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Project paths (all relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "diabd.csv")

FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

# ---------------------------------------------------------------------------
# Dataset-specific column definitions
# ---------------------------------------------------------------------------
# The target column in the DiaBD dataset. Values are the strings "Yes"/"No".
TARGET_COLUMN = "diabetic"

# Continuous numerical measurements.
NUMERICAL_FEATURES = [
    "age",
    "pulse_rate",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "height",
    "weight",
    "bmi",
]

# Categorical features. "gender" is text (Female/Male) and needs encoding.
# The rest are already stored as 0/1 integers in the raw CSV, so they only
# need to be treated as categorical (not scaled) rather than re-encoded.
BINARY_TEXT_FEATURE = "gender"
BINARY_FLAG_FEATURES = [
    "family_diabetes",
    "hypertensive",
    "family_hypertension",
    "cardiovascular_disease",
    "stroke",
]


def ensure_output_dirs():
    """Create the outputs/ subfolders if they do not already exist."""
    for directory in (FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        os.makedirs(directory, exist_ok=True)
