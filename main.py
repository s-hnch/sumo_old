"""
Sumo: scikit-learn regression pipeline for tabular data
Surrogate modeling pipeline - Main entry point

Usage:
    python main.py [data_file.csv]
"""

import sys
from src.data import load_data, split_data
from src.model import build_pipeline, train_pipeline
from src.evaluation import evaluate_model, get_residuals, print_residual_stats
from src.utils import save_model, save_results, get_feature_importance


def run_pipeline(data_path=None):
    """
    Run the complete modeling pipeline.
    
    Args:
        data_path: Optional path to CSV file
    """
    print("=" * 50)
    print("  Sumo: Regression Pipeline for Surrogate Modeling")
    print("=" * 50)
    
    try:
        # Step 1-2: Data loading, validation, and splitting
        print("\n[1-2] Loading and preparing data...")
        df = load_data(data_path)
        X_train, X_test, y_train, y_test, feature_names = split_data(df)
        
        # Step 3-6: Build and train pipeline
        print("\n[3-6] Building and training pipeline...")
        pipeline = build_pipeline(model_type="random_forest", preprocess=True)
        trained_pipeline = train_pipeline(pipeline, X_train, y_train)
        
        # Step 7-10: Evaluation
        print("\n[7-10] Evaluating model...")
        metrics = evaluate_model(trained_pipeline, X_test, y_test)
        
        # Step 9: Residual diagnostics
        residuals = get_residuals(trained_pipeline, X_test, y_test)
        print_residual_stats(residuals)
        
        # Step 11: Feature importance
        feat_imp = get_feature_importance(trained_pipeline)
        if feat_imp:
            print("\n  Feature Importance:")
            for i, imp in enumerate(feat_imp["importances"]):
                print(f"    {feature_names[i]}: {imp:.4f}")
        
        # Step 12: Save results
        print("\n[12] Saving results...")
        save_model(trained_pipeline)
        save_results(metrics)
        
        print("\n" + "=" * 50)
        print("  Pipeline completed successfully!")
        print("=" * 50)
        
        return metrics
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    # Get data path from command line if provided
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(data_path)
