"""
Utilities module
Step 11: Feature Importance Analysis (placeholder)
Step 12: Saving Model and Results
"""

import os
import joblib
import json
from config import MODELS_DIR, RESULTS_DIR


def save_model(pipeline, filepath=None, dirname=None):
    """
    Save a trained pipeline to disk.
    
    Args:
        pipeline: Trained scikit-learn pipeline
        filepath: Custom filepath. If None, uses default.
        dirname: Custom directory. If None, uses MODELS_DIR.
    
    Returns:
        str: Path where model was saved
    """
    if dirname is None:
        dirname = MODELS_DIR
    
    os.makedirs(dirname, exist_ok=True)
    
    if filepath is None:
        from config import MODEL_FILE
        filepath = os.path.join(dirname, MODEL_FILE)
    elif not os.path.isabs(filepath):
        filepath = os.path.join(dirname, filepath)
    
    joblib.dump(pipeline, filepath)
    print(f"\n  Model saved to: {filepath}")
    return filepath


def save_results(metrics, filepath=None, dirname=None):
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Custom filepath. If None, auto-generates.
        dirname: Custom directory. If None, uses RESULTS_DIR.
    
    Returns:
        str: Path where results were saved
    """
    if dirname is None:
        dirname = RESULTS_DIR
    
    os.makedirs(dirname, exist_ok=True)
    
    if filepath is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(dirname, f"results_{timestamp}.json")
    elif not os.path.isabs(filepath):
        filepath = os.path.join(dirname, filepath)
    
    # Add timestamp
    metrics["timestamp"] = str(datetime.datetime.now())
    
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  Results saved to: {filepath}")
    return filepath


def get_feature_importance(pipeline):
    """
    Extract feature importance from trained pipeline.
    
    Args:
        pipeline: Trained pipeline with a model that has feature_importances_
    
    Returns:
        dict or None: Feature importance dictionary, or None if not available
    """
    # Get the model from the pipeline
    model = pipeline.named_steps.get("model")
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Get feature names from preprocessor if available
        # For now, return as array (will improve with ColumnTransformer)
        return {"importances": importances.tolist()}
    
    return None
