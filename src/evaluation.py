"""
Evaluation module
Step 7: Cross Validation (placeholder)
Step 8: Evaluation Metrics
Step 9: Residual Diagnostics (placeholder)
Step 10: Uncertainty Quantification (placeholder)
"""

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
import numpy as np


def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate a trained pipeline on test data.
    
    Args:
        pipeline: Trained scikit-learn pipeline
        X_test: Test features
        y_test: Test target
    
    Returns:
        dict: Dictionary of metrics
    """
    predictions = pipeline.predict(X_test)
    
    metrics = {
        "mse": mean_squared_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "explained_variance": explained_variance_score(y_test, predictions),
    }
    
    print("\n  === Evaluation Metrics ===")
    print(f"  Mean Squared Error (MSE):  {metrics['mse']:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"  Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"  R-squared (R²):           {metrics['r2']:.4f}")
    print(f"  Explained Variance:       {metrics['explained_variance']:.4f}")
    
    return metrics


def get_residuals(pipeline, X_test, y_test):
    """
    Get residuals for diagnostic analysis.
    
    Args:
        pipeline: Trained pipeline
        X_test: Test features
        y_test: Test target
    
    Returns:
        np.array: Array of residuals (y_true - y_pred)
    """
    predictions = pipeline.predict(X_test)
    return y_test.values - predictions


def print_residual_stats(residuals):
    """Print basic residual statistics."""
    print("\n  === Residual Statistics ===")
    print(f"  Mean:   {np.mean(residuals):.4f}")
    print(f"  Std:    {np.std(residuals):.4f}")
    print(f"  Min:    {np.min(residuals):.4f}")
    print(f"  Max:    {np.max(residuals):.4f}")
