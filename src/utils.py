"""
Utilities module

STEP 11: Feature Importance Analysis
STEP 12: Saving Model and Results

This module provides helper functions for saving results and analyzing
 feature importance.
"""

import os
import joblib
import json
import datetime
from config import MODELS_DIR, RESULTS_DIR


def save_model(pipeline, filepath=None, dirname=None):
    """
    Save a trained pipeline to disk for later use.
    
    STEP 12: Saving Model. This allows you to reuse your trained model
    without having to retrain it every time.
    
    Args:
        pipeline: Trained scikit-learn pipeline.
        filepath: Custom filename. If None, uses default.
        dirname: Custom directory. If None, uses MODELS_DIR from config.
        
    Returns:
        str: Full path where model was saved.
        
    Example:
        >>> save_model(trained_pipeline)
        >>> # Model saved to models/regressor.joblib
    """
    if dirname is None:
        dirname = MODELS_DIR
    
    # Create directory if it doesn't exist
    os.makedirs(dirname, exist_ok=True)
    
    if filepath is None:
        from config import MODEL_FILE
        filepath = os.path.join(dirname, MODEL_FILE)
    elif not os.path.isabs(filepath):
        filepath = os.path.join(dirname, filepath)
    
    # Save using joblib (good for scikit-learn models)
    joblib.dump(pipeline, filepath)
    print(f"\n  Model saved to: {filepath}")
    print(f"    (You can load it later with: joblib.load('{filepath}'))")
    return filepath


def save_results(metrics, filepath=None, dirname=None):
    """
    Save evaluation metrics to a JSON file.
    
    STEP 12: Saving Results. This saves your evaluation metrics so you
    can compare different models or keep a record of performance.
    
    Args:
        metrics: Dictionary of metrics (from evaluate_model).
        filepath: Custom filename. If None, auto-generates with timestamp.
        dirname: Custom directory. If None, uses RESULTS_DIR from config.
        
    Returns:
        str: Full path where results were saved.
        
    Example:
        >>> save_results(metrics)
        >>> # Results saved to results/results_20240115_143022.json
    """
    if dirname is None:
        dirname = RESULTS_DIR
    
    # Create directory if it doesn't exist
    os.makedirs(dirname, exist_ok=True)
    
    if filepath is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(dirname, f"results_{timestamp}.json")
    elif not os.path.isabs(filepath):
        filepath = os.path.join(dirname, filepath)
    
    # Add timestamp to metrics
    metrics["timestamp"] = str(datetime.datetime.now())
    
    # Save as JSON (human-readable format)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  Results saved to: {filepath}")
    print(f"    (Human-readable JSON file with all metrics)")
    return filepath


def get_feature_importance(pipeline):
    """
    Extract feature importance from a trained pipeline.
    
    STEP 11: Feature Importance Analysis. This tells you which input
    features (columns) are most important for making predictions.
    
    Note: Not all models provide feature importance. Currently works with
    tree-based models like RandomForestRegressor.
    
    Args:
        pipeline: Trained pipeline.
        
    Returns:
        dict or None: Dictionary with 'importances' key containing a list of
        importance values, or None if model doesn't support it.
        
        Higher values = more important features
        Values sum to 1.0 (or close to it)
        
    Example:
        >>> feat_imp = get_feature_importance(trained_pipeline)
        >>> if feat_imp:
        ...     print(f"Most important feature: {max(feat_imp['importances'])}")
    """
    # Get the model from the pipeline
    model = pipeline.named_steps.get("model")
    
    # Check if model has feature_importances_ attribute
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Return as list (can be converted to array if needed)
        return {"importances": importances.tolist()}
    
    # Model doesn't support feature importance
    print("    Note: This model type doesn't provide feature importance")
    return None
