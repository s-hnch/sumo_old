# Sumo: Surrogate Modeling Pipeline

A scikit-learn pipeline for building regression models from tabular data.

## Quick Start

### 1. Prepare your data

Put your data in `data/raw/input.csv` in this format:

```csv
temperature,pressure,efficiency
20,100,85
25,110,88
30,95,82
```

- **Each row** = one data point
- **Last column** = the value you want to predict (target)
- **Other columns** = input features (used for prediction)

### 2. Install dependencies

```bash
# Create virtual environment (only once)
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install required packages
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python main.py
```

### 4. Find your results

- **Trained model:** `models/regressor.joblib`
- **Evaluation metrics:** `results/results_*.json`

---

## Pipeline Flow

```
┌─────────────┐    ┌───────────────┐    ┌─────────────────┐
│   CSV File   │───▶│  STEP 1-2:    │───▶│  STEP 3:         │
│  (your data) │    │  Load &       │    │  Standardize    │
└─────────────┘    │  Validate     │    │  Features        │
                   └───────────────┘    └─────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────┐
│                  STEP 4-6: MODEL PIPELINE                 │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │ Standardize │───▶│ Train Model      │───▶│ Make        │ │
│  │ Features    │    │ (Random Forest)  │    │ Predictions │ │
│  └─────────────┘    └─────────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────┐
│                  STEP 7-12: EVALUATION                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │ Calculate   │    │ Check       │    │ Save Model &    │ │
│  │ Metrics     │───▶│ Residuals   │───▶│ Results         │ │
│  │ (MSE, R²)   │    │ (Diagnostics)│    │                 │ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Glossary

| Term | What it means | Example |
|------|---------------|---------|
| **Features** | Input variables used for prediction | temperature, pressure, speed |
| **Target** | The value you want to predict | efficiency, cost, output |
| **Pipeline** | A sequence of processing steps | Load → Clean → Train → Evaluate |
| **Train/Test Split** | Dividing data: some to learn from, some to test | 80% to train, 20% to test |
| **Standardize** | Scale features to similar ranges | Transform values so all features have mean=0, std=1 |
| **Model** | A mathematical function that makes predictions | Random Forest, Linear Regression |
| **Metrics** | Numbers that measure model performance | MSE, R² |
| **Residuals** | Prediction errors (actual - predicted) | If true=10, predicted=9, residual=1 |

---

## Project Structure

```
sumo/
├── main.py                # Run this! Pipeline entry point
├── config.py              # Settings and paths
├── requirements.txt       # Python packages needed
├── README.md              # This file
├── data/
│   └── raw/
│       └── input.csv      # PUT YOUR DATA HERE
├── models/                # Saved trained models
├── results/               # Evaluation results
└── src/
    ├── data.py            # Loading and splitting data
    ├── preprocessing.py   # Feature standardization
    ├── model.py           # Model building and training
    ├── evaluation.py      # Metrics and diagnostics
    └── utils.py           # Saving models and results
```

---

## Customization

### Use a different data file

```bash
python main.py path/to/your/data.csv
```

### Change settings

Edit `config.py` to modify:
- Test set size (`TEST_SIZE = 0.2`)
- Random seed for reproducibility (`RANDOM_STATE = 42`)
- Number of trees in Random Forest (`N_ESTIMATORS = 100`)
