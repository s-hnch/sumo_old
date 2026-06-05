"""
Evaluation module

STEP 7: Cross Validation (placeholder for future implementation)
STEP 8: Evaluation Metrics
STEP 9: Residual Diagnostics
STEP 10: Uncertainty Quantification (placeholder for future implementation)

This module handles evaluating how well your model performs.
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
    Calculate performance metrics for your trained model.
    
    STEP 8: Evaluation Metrics. This tells you how well your model
    predicts on data it hasn't seen before (the test set).
    
    Args:
        pipeline: Trained scikit-learn pipeline.
        X_test: Test features (input values for evaluation).
        y_test: True test target values (ground truth).
        
    Returns:
        dict: Dictionary containing various evaluation metrics:
            - mse: Mean Squared Error (lower is better)
            - rmse: Root Mean Squared Error (same units as target)
            - mae: Mean Absolute Error (same units as target)
            - r2: R-squared (1.0 = perfect, 0.0 = no better than average)
            - explained_variance: Proportion of variance explained
        
    Example:
        >>> metrics = evaluate_model(trained_pipeline, X_test, y_test)
        >>> print(f"R² score: {metrics['r2']:.3f}")
    """
    predictions = pipeline.predict(X_test)
    
    # Calculate various metrics
    metrics = {
        "mse": mean_squared_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "explained_variance": explained_variance_score(y_test, predictions),
    }
    
    print("\n  === Evaluation Metrics ===")
    print(f"  Mean Squared Error (MSE):  {metrics['mse']:.4f}")
    print(f"    (Average squared prediction error - lower is better)")
    print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"    (Average prediction error in target units - lower is better)")
    print(f"  Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"    (Average absolute prediction error - lower is better)")
    print(f"  R-squared (R²):           {metrics['r2']:.4f}")
    print(f"    (0.0 = no better than guessing average, 1.0 = perfect)")
    print(f"  Explained Variance:       {metrics['explained_variance']:.4f}")
    print(f"    (Proportion of target variance captured by model)")
    
    return metrics


def get_residuals(pipeline, X_test, y_test):
    """
    Calculate prediction residuals (errors).
    
    STEP 9: Residual Diagnostics. Residuals are the differences between
    the true values and the predicted values. Analyzing residuals helps
    you understand where your model makes errors.
    
    Args:
        pipeline: Trained pipeline.
        X_test: Test features.
        y_test: True test target values.
        
    Returns:
        np.array: Array of residuals (y_true - y_pred).
        Positive residual = model predicted too low
        Negative residual = model predicted too high
    """
    predictions = pipeline.predict(X_test)
    return y_test.values - predictions


def print_residual_stats(residuals):
    """
    Print basic statistics about residuals.
    
    This gives you a quick overview of your model's errors.
    
    Args:
        residuals: Array of residuals from get_residuals().
    """
    print("\n  === Residual Statistics ===")
    print(f"  Mean:   {np.mean(residuals):.4f}")
    print(f"    (Ideal: close to 0. Positive = model under-predicts on average)")
    print(f"  Std:    {np.std(residuals):.4f}")
    print(f"    (Spread of errors. Lower = more consistent predictions)")
    print(f"  Min:    {np.min(residuals):.4f}")
    print(f"    (Worst under-prediction)")
    print(f"  Max:    {np.max(residuals):.4f}")
    print(f"    (Worst over-prediction)")
