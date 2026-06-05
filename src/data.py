"""
Data loading, validation, and splitting module

STEP 1: Data loading and validation
STEP 2: Train-Test Split

This module handles everything related to getting your data ready for modeling.
Think of it as preparing your ingredients before cooking.
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
    Load your CSV data file and validate it.
    
    This is like opening an Excel file and checking that it has the data
    you expect before you start analyzing it.
    
    Args:
        filepath: Path to your CSV file. If None, uses the default location.
        
    Returns:
        pd.DataFrame: Your data as a table that Python can work with.
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If the data file is empty or invalid.
    
    Example:
        >>> df = load_data()  # Loads data/raw/input.csv
        >>> df = load_data("my_data.csv")  # Loads specific file
    """
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, DEFAULT_DATA_FILE)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found at {filepath}. "
            f"Please add your CSV file to {RAW_DATA_DIR}/"
        )
    
    # Read the CSV file into a DataFrame (a table-like structure)
    df = pd.read_csv(filepath)
    
    # Validate the data to make sure it's usable
    validate_data(df, filepath)
    
    return df


def validate_data(df, filepath=None):
    """
    Check that your data is valid before using it for modeling.
    
    This performs several checks to catch common problems early:
    - Is the file empty?
    - Are there enough columns?
    - Are any columns completely empty?
    
    Args:
        df: The DataFrame (table) to validate.
        filepath: The original file path (used for error messages).
        
    Raises:
        ValueError: If any validation check fails.
    """
    # Check the file isn't empty
    if len(df) == 0:
        raise ValueError(f"Data file {filepath} is empty (0 rows)")
    
    # Check we have at least one feature and one target
    if len(df.columns) < REQUIRED_COLUMNS_MIN:
        raise ValueError(
            f"Data must have at least {REQUIRED_COLUMNS_MIN} columns "
            f"(one or more features + one target), found {len(df.columns)}"
        )
    
    # Check for columns that are completely empty (all NaN/None)
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        raise ValueError(f"Columns with all missing values: {empty_cols}")
    
    print(f"✓ Data validated: {len(df)} rows, {len(df.columns)} columns")


def split_data(df, target_column=None, test_size=None, random_state=None):
    """
    Split your data into training and testing sets.
    
    This does three things:
    1. Separates features (inputs) from target (output to predict)
    2. Selects only numeric columns (for now)
    3. Splits into training data (to learn from) and test data (to evaluate)
    
    Think of it like: you study 80% of your notes (training) and then
    test yourself on the remaining 20% (testing) to see how well you learned.
    
    Args:
        df: Your data as a DataFrame.
        target_column: Name of the column you want to predict.
            If None, uses the last column.
        test_size: Fraction of data to use for testing (e.g., 0.2 = 20%).
            If None, uses the default from config.py.
        random_state: Seed for random number generator (for reproducibility).
            If None, uses the default from config.py.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
        
        - X_train: Training features (input values for training)
        - X_test: Test features (input values for evaluation)
        - y_train: Training target (output values for training)
        - y_test: Test target (output values for evaluation)
        - feature_names: List of feature column names
    """
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE
    if target_column is None:
        target_column = df.columns[-1]
    
    # Separate features (X) from target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # For now, only use numeric columns (we'll add categorical support later)
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X = X[numeric_cols]
    
    feature_names = list(numeric_cols)
    
    print(f"  Features: {feature_names}")
    print(f"  Target: {target_column}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names
