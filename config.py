"""
Configuration for Sumo pipeline
All constants and paths in one place for easy modification
"""

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Default filenames
DEFAULT_DATA_FILE = "input.csv"
MODEL_FILE = "regressor.joblib"

# Data splitting
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model settings
N_ESTIMATORS = 100

# Validation
REQUIRED_COLUMNS_MIN = 2  # At least 1 feature + 1 target
