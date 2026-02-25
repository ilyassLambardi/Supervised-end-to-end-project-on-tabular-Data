# Supervised End-to-End Project on Tabular Data

An end-to-end supervised machine learning project for predicting Boston housing prices (`MEDV`). The pipeline covers exploratory data analysis, feature engineering, baseline & advanced model training, hyperparameter tuning, model evaluation, and interpretation.

## Project Structure

```
├── data/
│   └── housing.csv                # Boston Housing dataset (506 × 14)
├── notebooks/
│   └── data_EDA.ipynb             # Exploratory Data Analysis
├── src/
│   ├── data_load.py               # Data loading utilities
│   ├── baseline_data_preparation.py  # Preprocessing pipeline & baseline models
│   ├── model_training.ipynb       # Advanced model training & tuning
│   ├── evaluation.ipynb           # Model evaluation & comparison
│   └── interpretation.ipynb       # SHAP-based model interpretation
├── models/
│   └── final_stack_model.pkl      # Saved final stacking ensemble
├── outputs/
│   └── data_preparation_results/  # Preprocessed data & evaluation CSVs
├── environment.yml                # Conda environment specification
├── requirements.txt               # pip dependencies
└── README.md
```

## Dataset

The Boston Housing dataset contains 506 samples with 13 input features and 1 target variable:

| Feature | Description |
|---------|-------------|
| CRIM | Per capita crime rate |
| ZN | Proportion of residential land zoned for large lots |
| INDUS | Proportion of non-retail business acres |
| CHAS | Charles River dummy variable (1 if tract bounds river) |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built before 1940 |
| DIS | Weighted distances to employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Property tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio |
| B | 1000(Bk − 0.63)², where Bk is the proportion of Black residents |
| LSTAT | Percentage of lower status population |
| **MEDV** | **Median value of owner-occupied homes (target)** |

## Pipeline Overview

### 1. Exploratory Data Analysis
- Distribution analysis, correlation heatmaps, missing value checks (`notebooks/data_EDA.ipynb`)

### 2. Data Preparation
- **Log transform** → RobustScaler: `CRIM`, `ZN`, `NOX`, `DIS`, `LSTAT`
- **Yeo-Johnson** → RobustScaler: `B`
- **RobustScaler** only: `RM`
- **Passthrough** (binary): `CHAS`
- **OneHotEncoder**: `RAD`
- **Binarization** (X > 0): `ZN`
- Train / Validation / Test split (70% / 15% / 15%)

### 3. Baseline Models
- Linear Regression
- Decision Tree Regressor
- MLP Regressor

### 4. Advanced Models & Tuning
- XGBoost, LightGBM, CatBoost
- Hyperparameter optimization with **Optuna**
- **Stacking ensemble** as the final model

### 5. Model Interpretation
- SHAP values for feature importance and model explanations

## Getting Started

### Prerequisites
- Python 3.12+ / Conda

### Installation

**Using Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate torch_envv
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### Running the Project

1. **EDA** — Open and run `notebooks/data_EDA.ipynb`
2. **Data Preparation & Baselines** — Run `src/baseline_data_preparation.py`
3. **Model Training** — Open and run `src/model_training.ipynb`
4. **Evaluation** — Open and run `src/evaluation.ipynb`
5. **Interpretation** — Open and run `src/interpretation.ipynb`

## Key Libraries

| Library | Purpose |
|---------|---------|
| scikit-learn | Preprocessing, baseline models, stacking |
| XGBoost / LightGBM / CatBoost | Gradient boosting models |
| Optuna | Hyperparameter optimization |
| SHAP | Model interpretation |
| matplotlib / seaborn | Visualization |
| pandas / NumPy | Data manipulation |

## Author

**Ilyass Lambardi**
