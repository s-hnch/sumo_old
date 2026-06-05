"""
Model building and training module

STEP 4: Kernel Definition (placeholder for future Gaussian Process models)
STEP 5: Hyperparameter Optimization (placeholder for future)
STEP 6: Model Training

This module handles building and training machine learning models.
"""

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from config import N_ESTIMATORS, RANDOM_STATE
from src.preprocessing import build_preprocessor


def build_pipeline(model_type="random_forest", preprocess=True):
    """
    Build the complete machine learning pipeline.
    
    A pipeline is a sequence of steps that data flows through:
    1. Preprocessing (e.g., standardization) - STEP 3
    2. Model training - STEP 6
    
    This ensures that preprocessing is applied correctly during both
    training and prediction.
    
    Args:
        model_type: Type of model to use. Options:
            - "random_forest": Random Forest Regressor (default, good for most cases)
            - "linear": Linear Regression (simpler, faster)
        preprocess: Whether to include feature standardization (True/False).
        
    Returns:
        Pipeline: A scikit-learn Pipeline object.
        
    Example:
        >>> pipeline = build_pipeline(model_type="random_forest")
        >>> # This creates: StandardScaler -> RandomForestRegressor
    """
    steps = []
    
    # STEP 3: Preprocessing
    if preprocess:
        preprocessor = build_preprocessor()
        steps.append(("preprocessor", preprocessor))
    
    # STEP 6: Model (Step 4-5: Kernel definition and hyperparameter tuning are
    # placeholders for future Gaussian Process implementation)
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE
        )
        print("  Model: RandomForestRegressor")
        print(f"    - Uses {N_ESTIMATORS} decision trees")
        print(f"    - Each tree votes, final prediction is the average")
    elif model_type == "linear":
        model = LinearRegression()
        print("  Model: LinearRegression")
        print(f"    - Finds linear relationships between features and target")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    steps.append(("model", model))
    
    pipeline = Pipeline(steps)
    
    print("  Pipeline steps:")
    for name, step in pipeline.steps:
        print(f"    - {name}: {step.__class__.__name__}")
    
    return pipeline


def train_pipeline(pipeline, X_train, y_train):
    """
    Train the pipeline on your training data.
    
    This is STEP 6: Model Training. The pipeline learns patterns from your
    training data that allow it to make predictions on new, unseen data.
    
    Args:
        pipeline: A scikit-learn Pipeline (from build_pipeline).
        X_train: Training features (input values).
        y_train: Training target (output values to predict).
        
    Returns:
        Pipeline: The trained pipeline, ready to make predictions.
        
    Example:
        >>> trained_pipeline = train_pipeline(pipeline, X_train, y_train)
        >>> predictions = trained_pipeline.predict(X_new)
    """
    print("\n  Training model...")
    pipeline.fit(X_train, y_train)
    print("  Training complete")
    return pipeline
