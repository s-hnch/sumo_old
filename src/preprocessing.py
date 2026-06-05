"""
Feature preprocessing module
Step 3: Feature Standardization
"""

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
import numpy as np


def build_preprocessor(feature_names=None, scale=True):
    """
    Build preprocessing pipeline for features.
    
    Args:
        feature_names: List of feature names (for logging)
        scale: Whether to apply standardization
    
    Returns:
        ColumnTransformer or Pipeline: Preprocessing transformer
    """
    steps = []
    
    if scale:
        print("  Applying StandardScaler to all numeric features")
        steps.append(("scaler", StandardScaler()))
    
    # For now, simple pipeline. Will expand for:
    # - Different preprocessing per feature type
    # - Kernel transformations
    # - Feature engineering
    
    if len(steps) == 1:
        return steps[0][1]  # Return just the transformer
    elif len(steps) > 1:
        return SklearnPipeline(steps)
    else:
        return "passthrough"  # No preprocessing


def get_numeric_transformer():
    """Get transformer for numeric features."""
    return StandardScaler()


def get_categorical_transformer():
    """Get transformer for categorical features (placeholder for future)."""
    # Will add OneHotEncoder or OrdinalEncoder when needed
    return FunctionTransformer(lambda x: x, validate=False)  # Identity
