"""
Data loading, cleaning, and preprocessing for the DiaBD dataset.

This module is intentionally split into small, single-purpose functions:
    load_data()          -> read the raw CSV
    clean_data()          -> remove confirmed bad rows (documented, not silent)
    encode_target()       -> turn "Yes"/"No" into 1/0
    split_data()           -> stratified 80/20 train-test split
    build_preprocessor()  -> ColumnTransformer (scaling + one-hot encoding)

Keeping cleaning and splitting separate from scaling/encoding is important
for avoiding data leakage: the ColumnTransformer built in
build_preprocessor() is only ever *fit* on the training set (this happens
inside the modelling Pipelines in train_models.py), never on the full
dataset.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    BINARY_FLAG_FEATURES,
    BINARY_TEXT_FEATURE,
    DATA_PATH,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw DiaBD CSV into a DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove rows containing physiologically impossible values.

    These were identified during manual inspection of the raw dataset
    (see report/final_report.md, Section 3 "Dataset Description"):

      - glucose == 0 mmol/L        -> incompatible with a living patient
      - height < 1.0 m (adult)     -> impossible adult height, also causes
                                       BMI values as high as 574 (since
                                       bmi = weight / height^2 in this data)
      - pulse_rate < 30 bpm        -> incompatible with life

    This is NOT a blanket outlier-removal step. Plausible-but-extreme
    values (e.g. very low blood pressure, short-but-real stature) are left
    untouched; only values that are biologically impossible are removed.
    Duplicate rows are also checked and removed if present.
    """
    df = df.copy()
    n_start = len(df)

    impossible_mask = (
        (df["glucose"] == 0)
        | (df["height"] < 1.0)
        | (df["pulse_rate"] < 30)
    )
    n_impossible = int(impossible_mask.sum())
    df = df.loc[~impossible_mask].reset_index(drop=True)

    n_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    if verbose:
        print(f"Rows before cleaning: {n_start}")
        print(f"Rows removed (impossible values): {n_impossible}")
        print(f"Duplicate rows removed: {n_duplicates}")
        print(f"Rows after cleaning: {len(df)}")

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map the target column 'diabetic' from Yes/No text to 1/0 integers."""
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return df


def split_data(df: pd.DataFrame):
    """
    Separate features/target and perform an 80/20 stratified split.

    Stratifying on y keeps the same diabetic/non-diabetic ratio in both
    the training and test sets, which matters a lot given the severe
    class imbalance in this dataset.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - scales numerical features with StandardScaler (only if scale_numeric)
      - one-hot encodes the 'gender' text column
      - passes the already-binary 0/1 flag columns through unchanged

    scale_numeric=True is used for Logistic Regression and KNN, which are
    distance/gradient based and sensitive to feature scale.
    scale_numeric=False is used for Random Forest and XGBoost, which split
    on raw feature thresholds and are unaffected by monotonic scaling.

    Because this transformer is placed inside an sklearn Pipeline and only
    ever called with .fit() on X_train, the scaler's mean/std and the
    one-hot categories are learned strictly from the training data, so
    there is no leakage of test-set information.
    """
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("gender", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), [BINARY_TEXT_FEATURE]),
            ("flags", "passthrough", BINARY_FLAG_FEATURES),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer):
    """
    Retrieve human-readable feature names after the ColumnTransformer has
    been fitted. Used later for feature importance plots, since one-hot
    encoding changes/adds column names (e.g. 'gender' -> 'gender_Male').
    """
    return list(preprocessor.get_feature_names_out())
