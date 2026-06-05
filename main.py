"""
Sumo: scikit-learn regression pipeline for tabular data
Surrogate modeling pipeline
"""

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import joblib
import os

# Configuration
DATA_PATH = "data/raw/input.csv"
MODEL_SAVE_PATH = "models/regressor.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(filepath=DATA_PATH):
    """Load tabular data from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_data(df, target_column):
    """
    Prepare data for regression.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of the column to predict
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Select only numeric columns
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X = X[numeric_cols]
    
    print(f"Using {len(numeric_cols)} numeric features: {list(numeric_cols)}")
    print(f"Target variable: {target_column}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_pipeline():
    """Build the scikit-learn pipeline."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE
        ))
    ])
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the trained pipeline."""
    predictions = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n=== Model Evaluation ===")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    return {"mse": mse, "mae": mae, "r2": r2}


def save_model(pipeline, filepath=MODEL_SAVE_PATH):
    """Save the trained pipeline."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"\nModel saved to {filepath}")


def main():
    """Run the complete pipeline."""
    print("=== Sumo: Regression Pipeline ===\n")
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        print("Please add your tabular data CSV file to data/raw/input.csv")
        print("\nExample format:")
        print("feature1,feature2,feature3,target")
        print("1.5,2.3,0.8,10.2")
        print("2.1,3.7,1.2,15.6")
        return
    
    # Step 1: Load data
    df = load_data()
    
    # Get target column (assume last column is target)
    target_column = df.columns[-1]
    
    # Step 2: Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, target_column)
    
    # Step 3: Build pipeline
    pipeline = build_pipeline()
    print("\nPipeline steps:")
    for name, step in pipeline.steps:
        print(f"  - {name}: {step.__class__.__name__}")
    
    # Step 4: Train
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    
    # Step 5: Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # Step 6: Save
    save_model(pipeline)
    
    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
