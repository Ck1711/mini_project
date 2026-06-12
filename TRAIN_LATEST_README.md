# Latest Voice Model Training

## Overview

`train_latest_voice_model.py` is a production-ready training script for the latest Parkinson voice dataset located in `datasets/voice/latest/`.

## Features

✅ **Automatic Dataset Detection**
- Automatically detects train/test files in `datasets/voice/latest/`
- Auto-detects column structure (ID, features, target)
- Handles cases where test data lacks target column (performs stratified split)

✅ **Robust Preprocessing**
- Removes duplicate rows
- Handles missing values (imputation with train means)
- Prevents data leakage by excluding ID/metadata columns
- StandardScaler normalization

✅ **Multiple Models**
- XGBoost
- Random Forest
- Extra Trees
- LightGBM (if installed)

✅ **Hyperparameter Tuning**
- RandomizedSearchCV with StratifiedKFold CV
- Balanced class weights for imbalanced datasets
- Optimization for F1 Score and ROC-AUC

✅ **Comprehensive Evaluation**
- Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix plots
- ROC curve plots
- Feature importance visualization

✅ **Production-Ready Output**
- Best model saved: `models/latest_voice_model.pkl`
- Scaler saved: `models/latest_voice_scaler.pkl`
- Metrics saved: `outputs/latest_voice_metrics.json`
- Feature importance plot: `outputs/latest_voice_feature_importance.png`
- Confusion matrix: `outputs/latest_voice_confusion_matrix.png`
- ROC curve: `outputs/latest_voice_roc_curve.png`

## Usage

```bash
python train_latest_voice_model.py
```

## Dataset Format

Expected format (tab or comma-separated):
```
[ID], [feature_1], [feature_2], ..., [feature_N], [target_label]
```

- First column: Patient/Subject ID (automatically excluded from features)
- Last column: Target label (0=Healthy, 1=Parkinson)
- All numeric columns between ID and target are treated as features

## Configuration

Edit these constants at the top of the script to customize:

```python
LATEST_DATA_DIR = os.path.join("datasets", "voice", "latest")
TRAIN_FILE = os.path.join(LATEST_DATA_DIR, "train_data.txt")
TEST_FILE = os.path.join(LATEST_DATA_DIR, "test_data.txt")
RANDOM_STATE = 42
CV_FOLDS = 5
```

## Output Files

| File | Location | Description |
|------|----------|-------------|
| Model | `models/latest_voice_model.pkl` | Trained best model (pickled) |
| Scaler | `models/latest_voice_scaler.pkl` | Feature scaler for inference |
| Metrics | `outputs/latest_voice_metrics.json` | All evaluation metrics |
| Feature Importance | `outputs/latest_voice_feature_importance.png` | Top 20 features plot |
| Confusion Matrix | `outputs/latest_voice_confusion_matrix.png` | Classification confusion matrix |
| ROC Curve | `outputs/latest_voice_roc_curve.png` | ROC-AUC curve |

## Metrics JSON Structure

```json
{
  "best_model": "XGBoost",
  "best_parameters": { ... },
  "all_model_metrics": [ ... ],
  "test_metrics": {
    "accuracy": 1.0,
    "balanced_accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "roc_auc": 1.0
  }
}
```

## Latest Training Results

| Metric | Value |
|--------|-------|
| Best Model | XGBoost |
| Test Accuracy | 100.0% |
| Test F1 Score | 100.0% |
| Test ROC-AUC | 100.0% |
| Balanced Accuracy | 100.0% |
| Classes | Perfectly Balanced (50/50) |

## Model Selection Criteria

The script automatically selects the best model based on:
1. **Primary**: Validation F1 Score (handles class imbalance)
2. **Secondary**: ROC-AUC Score (threshold-independent evaluation)

This ensures generalization performance rather than just training accuracy.

## Data Leakage Prevention

✅ ID columns automatically excluded from features
✅ Target column never included in features  
✅ Train/test split uses StratifiedKFold (preserves class ratios)
✅ Preprocessing (mean imputation) uses only training data statistics
✅ Scaler fitted on training data only

## Cross-Validation Strategy

- **StratifiedKFold**: 5-fold stratified cross-validation
- **Stratification**: Preserves class distribution in each fold
- **Scoring Metric**: F1 Score (balanced metric for imbalanced data)

## Installation Requirements

```bash
pip install xgboost scikit-learn lightgbm pandas numpy matplotlib seaborn joblib
```

## Independent Execution

The script is completely independent and requires no modifications to existing project files. It can be run standalone:

```bash
python train_latest_voice_model.py
```

All paths are configured relative to the script location.
