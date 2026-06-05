"""
Model building and training module
Step 4: Kernel Definition (placeholder)
Step 5: Hyperparameter Optimization (placeholder)
Step 6: Model Training
"""

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from config import N_ESTIMATORS, RANDOM_STATE
from src.preprocessing import build_preprocessor


def build_pipeline(model_type="random_forest", preprocess=True):
    """
    Build the scikit-learn pipeline.
    
    Args:
        model_type: Type of model ("random_forest", "linear", etc.)
        preprocess: Whether to include preprocessing
    
    Returns:
        Pipeline: Complete scikit-learn pipeline
    """
    steps = []
    
    # Preprocessing
    if preprocess:
        preprocessor = build_preprocessor()
        steps.append(("preprocessor", preprocessor))
    
    # Model selection
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE
        )
        print("  Model: RandomForestRegressor")
    elif model_type == "linear":
        model = LinearRegression()
        print("  Model: LinearRegression")
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
    Train the pipeline on training data.
    
    Args:
        pipeline: scikit-learn Pipeline
        X_train: Training features
        y_train: Training target
    
    Returns:
        Pipeline: Trained pipeline
    """
    print("\n  Training model...")
    pipeline.fit(X_train, y_train)
    print("  Training complete")
    return pipeline
