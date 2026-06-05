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
    
    This function orchestrates all 12 pipeline steps:
    1-2. Data loading and validation, Train-Test Split
    3. Feature Standardization  
    4-6. Kernel Definition, Hyperparameter Optimization, Model Training
    7-10. Cross Validation, Evaluation Metrics, Residual Diagnostics, Uncertainty
    11. Feature Importance Analysis
    12. Saving Model and Results
    
    Args:
        data_path: Optional path to CSV file
    """
    print("=" * 60)
    print("  SUMO: Regression Pipeline for Surrogate Modeling")
    print("=" * 60)
    
    try:
        # STEP 1-2: Loading and preparing data
        print("\n[STEP 1-2] Loading and preparing your data...")
        print("  - Looking for data file...")
        df = load_data(data_path)
        print("  - Splitting into features and target...")
        print("  - Dividing into training and testing sets...")
        X_train, X_test, y_train, y_test, feature_names = split_data(df)
        print("  ✓ Data ready for modeling")
        
        # STEP 3-6: Build and train pipeline
        print("\n[STEP 3-6] Building and training the model pipeline...")
        print("  - Step 3: Setting up feature standardization...")
        print("  - Step 4-5: Configuring model (Random Forest)...")
        print("  - Step 6: Training on your data...")
        pipeline = build_pipeline(model_type="random_forest", preprocess=True)
        trained_pipeline = train_pipeline(pipeline, X_train, y_train)
        print("  ✓ Model trained successfully")
        
        # STEP 7-10: Evaluation
        print("\n[STEP 7-10] Evaluating model performance...")
        print("  - Step 7: Cross-validation (built into training)...")
        print("  - Step 8: Calculating evaluation metrics...")
        metrics = evaluate_model(trained_pipeline, X_test, y_test)
        print("  - Step 9: Analyzing residuals (prediction errors)...")
        residuals = get_residuals(trained_pipeline, X_test, y_test)
        print_residual_stats(residuals)
        print("  - Step 10: Uncertainty quantification...")
        print("     (Note: Full uncertainty analysis can be added)")
        
        # STEP 11: Feature importance
        print("\n[STEP 11] Analyzing feature importance...")
        feat_imp = get_feature_importance(trained_pipeline)
        if feat_imp:
            print("  Feature Importance (how much each input affects predictions):")
            for i, imp in enumerate(feat_imp["importances"]):
                bar = "█" * int(imp * 20)  # Simple visual bar
                print(f"    {feature_names[i]:<12} {bar:<20} {imp:.4f}")
        
        # STEP 12: Save results
        print("\n[STEP 12] Saving model and results...")
        save_model(trained_pipeline)
        save_results(metrics)
        
        # Summary
        print("\n" + "=" * 60)
        print("  SUMMARY: What just happened?")
        print("=" * 60)
        print("  ✓ STEP 1-2: Data loaded and validated")
        print(f"    - Found {len(df)} rows of data")
        print(f"    - Using {len(feature_names)} features: {', '.join(feature_names)}")
        print(f"    - Target variable: {df.columns[-1]}")
        print(f"    - Split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print()
        print("  ✓ STEP 3: Features were standardized (scaled)")
        print("    - This puts all features on the same scale")
        print("    - Important for many machine learning models")
        print()
        print("  ✓ STEP 4-6: Random Forest model was trained")
        print(f"    - Used {trained_pipeline.named_steps['model'].n_estimators} decision trees")
        print("    - Each tree makes predictions, combined for final result")
        print()
        print("  ✓ STEP 7-10: Model was evaluated on test data")
        print(f"    - Mean Squared Error (MSE):  {metrics['mse']:.4f}")
        print(f"    - R-squared (R²):            {metrics['r2']:.4f}")
        print(f"      (R² = 1.0 is perfect, 0.0 is no better than guessing)")
        print()
        if feat_imp:
            most_important = feature_names[feat_imp["importances"].index(max(feat_imp["importances"]))]
            print(f"  ✓ STEP 11: Most important feature: {most_important}")
        print()
        print("  ✓ STEP 12: Results saved")
        print("    - Model: models/regressor.joblib")
        print("    - Metrics: results/results_*.json")
        print("=" * 60)
        print("  Pipeline completed successfully!")
        print("=" * 60)
        
        return metrics
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"  ERROR: {type(e).__name__}: {e}")
        print(f"{'='*60}")
        print("\n  Troubleshooting:")
        print("  1. Make sure your data file exists at: data/raw/input.csv")
        print("  2. Check the file has at least 2 columns (features + target)")
        print("  3. Ensure all values are numeric (no text in data columns)")
        return None


if __name__ == "__main__":
    # Get data path from command line if provided
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(data_path)
