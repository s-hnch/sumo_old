"""
Data loading, validation, and splitting module
Step 1: Data loading and validation
Step 2: Train-Test Split
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    RAW_DATA_DIR,
    DEFAULT_DATA_FILE,
    TEST_SIZE,
    RANDOM_STATE,
    REQUIRED_COLUMNS_MIN,
)


def load_data(filepath=None):
    """
    Load tabular data from CSV file.
    
    Args:
        filepath: Path to CSV file. If None, uses default path.
    
    Returns:
        pd.DataFrame: Loaded data
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is invalid
    """
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, DEFAULT_DATA_FILE)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found at {filepath}. "
            f"Please add your CSV file to {RAW_DATA_DIR}/"
        )
    
    df = pd.read_csv(filepath)
    
    # Validate data
    validate_data(df, filepath)
    
    return df


def validate_data(df, filepath=None):
    """
    Validate the loaded dataframe.
    
    Args:
        df: DataFrame to validate
        filepath: Original filepath (for error messages)
    
    Raises:
        ValueError: If validation fails
    """
    # Check not empty
    if len(df) == 0:
        raise ValueError(f"Data file {filepath} is empty (0 rows)")
    
    # Check minimum columns
    if len(df.columns) < REQUIRED_COLUMNS_MIN:
        raise ValueError(
            f"Data must have at least {REQUIRED_COLUMNS_MIN} columns "
            f"(features + target), found {len(df.columns)}"
        )
    
    # Check for all-NaN columns
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        raise ValueError(f"Columns with all missing values: {empty_cols}")
    
    print(f"✓ Data validated: {len(df)} rows, {len(df.columns)} columns")


def split_data(df, target_column=None, test_size=None, random_state=None):
    """
    Split data into features and target, then into train/test sets.
    
    Args:
        df: DataFrame with data
        target_column: Name of target column. If None, uses last column.
        test_size: Fraction for test set. If None, uses config default.
        random_state: Random seed. If None, uses config default.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE
    if target_column is None:
        target_column = df.columns[-1]
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Select only numeric columns for now
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X = X[numeric_cols]
    
    feature_names = list(numeric_cols)
    
    print(f"  Features: {feature_names}")
    print(f"  Target: {target_column}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names
