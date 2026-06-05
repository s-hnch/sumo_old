"""
Feature preprocessing module

STEP 3: Feature Standardization

This module handles transforming your features (input variables) to make
 them suitable for machine learning models. Many models work better when
 all features are on similar scales.
"""

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
import numpy as np


def build_preprocessor(feature_names=None, scale=True):
    """
    Build preprocessing steps for your features.
    
    Standardization (Step 3) transforms your features so they have:
    - Mean (average) = 0
    - Standard deviation = 1
    
    This is like converting all measurements to the same unit system.
    
    Args:
        feature_names: List of feature names (for logging/information).
        scale: Whether to apply standardization (True/False).
        
    Returns:
        A scikit-learn transformer that can be used in a Pipeline.
        
    Example:
        >>> preprocessor = build_preprocessor(scale=True)
        >>> # This will standardize all numeric features
    """
    steps = []
    
    if scale:
        print("  Applying StandardScaler to all numeric features")
        print("    (This centers features around 0 and scales to unit variance)")
        steps.append(("scaler", StandardScaler()))
    
    # For now, simple pipeline. Future enhancements:
    # - Different preprocessing per feature type (numeric vs categorical)
    # - Kernel transformations for Gaussian Processes
    # - Feature engineering (creating new features from existing ones)
    
    if len(steps) == 1:
        return steps[0][1]  # Return just the transformer
    elif len(steps) > 1:
        return SklearnPipeline(steps)
    else:
        return "passthrough"  # No preprocessing


def get_numeric_transformer():
    """
    Get a transformer for numeric features.
    
    Returns:
        StandardScaler: Transformer that standardizes numeric features.
    """
    return StandardScaler()


def get_categorical_transformer():
    """
    Get a transformer for categorical features.
    
    Note: This is a placeholder for future implementation.
    Currently returns an identity transformer (does nothing).
    
    Returns:
        FunctionTransformer: Identity transformer (for now).
    """
    # Future: Add OneHotEncoder or OrdinalEncoder when needed
    return FunctionTransformer(lambda x: x, validate=False)
