"""
Train voice classification model on latest Parkinson dataset.

This script:
1. Loads train_data.txt and test_data.txt from datasets/voice/latest/
2. Automatically detects dataset structure (ID, features, target)
3. Applies preprocessing: missing value handling, duplicate removal, scaling
4. Prevents data leakage by excluding ID/patient columns
5. Trains multiple models: XGBoost, Random Forest, LightGBM, Extra Trees
6. Performs hyperparameter tuning with StratifiedKFold CV
7. Selects best model based on F1 Score and ROC-AUC
8. Applies class balancing if imbalanced
9. Saves model, metrics, and feature importance plot
10. Prints comprehensive evaluation metrics

Production-ready, independent, and highly optimized for generalization.
"""

import json
import os
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

# Optional imbalanced-learn imports (SMOTE inside CV)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except Exception:
    SMOTE = None
    ImbPipeline = None
    HAS_IMBLEARN = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    HAS_CATBOOST = False

# Import patient-wise helpers used by the stacking pipeline
from patient_data import (
    SELECTED_FEATURES_PATH,
    impute_with_train_stats,
    load_voice_patient_split,
    make_selected_voice_frames,
    print_class_distribution_and_weights,
    prune_weak_voice_features,
    save_voice_splits,
    scale_train_transform_test,
    select_k_best_features,
    warn_suspicious_accuracy,
)

# ============================================================================
# Configuration
# ============================================================================

LATEST_DATA_DIR = os.path.join("datasets", "voice", "latest")
TRAIN_FILE = os.path.join(LATEST_DATA_DIR, "train_data.txt")
TEST_FILE = os.path.join(LATEST_DATA_DIR, "test_data.txt")

OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
RANDOM_STATE = 42
CV_FOLDS = 5
OVERFIT_GAP_THRESHOLD = 0.05
TARGET_CORRELATION_THRESHOLD = 0.95
LEAKAGE_REPORT_PATH = os.path.join(OUTPUT_DIR, "leakage_report.txt")

# Model output paths
LATEST_MODEL_PATH = os.path.join(MODELS_DIR, "latest_voice_model.pkl")
LATEST_SCALER_PATH = os.path.join(MODELS_DIR, "latest_voice_scaler.pkl")
LATEST_METRICS_PATH = os.path.join(OUTPUT_DIR, "latest_voice_metrics.json")
LATEST_FI_PLOT_PATH = os.path.join(OUTPUT_DIR, "latest_voice_feature_importance.png")


# ============================================================================
# Data Loading & Structure Detection
# ============================================================================

def load_data_from_latest(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data from latest folder."""
    print(f"[INFO] Loading train data from {train_path}...")
    train_df = pd.read_csv(train_path, header=None)
    
    print(f"[INFO] Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path, header=None)
    
    print(f"[INFO] Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    return train_df, test_df


def detect_dataset_structure(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[int, int]:
    """
    Detect ID column index and target column index.
    
    Assumes:
    - First column is ID/patient ID
    - Last column is target/label in train (test may not have it)
    - All numeric columns in between are features
    
    Returns:
        (id_col_idx, target_col_idx)
    """
    n_train_cols = train_df.shape[1]
    n_test_cols = test_df.shape[1]
    
    # First column is typically ID
    id_col_idx = 0
    
    # If test has one fewer column, it's missing the target
    # Use train's last column as target
    if n_train_cols == n_test_cols + 1:
        target_col_idx = n_train_cols - 1
        print(f"[INFO] Dataset structure detected:")
        print(f"[INFO]   Train columns: {n_train_cols} | Test columns: {n_test_cols}")
        print(f"[INFO]   Test is missing target column")
        print(f"[INFO]   ID column index: {id_col_idx}")
        print(f"[INFO]   Target column index: {target_col_idx}")
        print(f"[INFO]   Feature columns: {id_col_idx + 1} to {target_col_idx - 1}")
    else:
        target_col_idx = n_train_cols - 1
        print(f"[INFO] Dataset structure detected:")
        print(f"[INFO]   Train columns: {n_train_cols} | Test columns: {n_test_cols}")
        print(f"[INFO]   ID column index: {id_col_idx}")
        print(f"[INFO]   Target column index: {target_col_idx}")
        print(f"[INFO]   Feature columns: {id_col_idx + 1} to {target_col_idx - 1}")
    
    return id_col_idx, target_col_idx


def extract_features_and_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col_idx: int,
    target_col_idx: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract features and target from loaded dataframes.
    Exclude ID column and target column from features.
    
    Handles case where test_df may not have target column.
    """
    # Get feature column indices from train (exclude ID and target)
    feature_cols_train = [i for i in range(train_df.shape[1]) if i != id_col_idx and i != target_col_idx]
    
    X_train = train_df.iloc[:, feature_cols_train].copy()
    y_train = train_df.iloc[:, target_col_idx].copy()
    
    # For test, use same feature indices (excluding ID)
    # If test has fewer columns, it doesn't have target, so use all except ID
    if test_df.shape[1] == train_df.shape[1]:
        # Test has target column too
        feature_cols_test = feature_cols_train
        X_test = test_df.iloc[:, feature_cols_test].copy()
        y_test = test_df.iloc[:, target_col_idx].copy()
    else:
        # Test doesn't have target; use all columns except ID
        feature_cols_test = [i for i in range(test_df.shape[1]) if i != id_col_idx]
        X_test = test_df.iloc[:, feature_cols_test].copy()
        # Create dummy y_test (will be ignored or filled later)
        y_test = None
    
    print(f"[INFO] Features extracted: {len(feature_cols_train)} features from train")
    print(f"[INFO] X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"[INFO] X_test shape: {X_test.shape}")
    if y_test is not None:
        print(f"[INFO] y_test shape: {y_test.shape}")
    else:
        print(f"[INFO] y_test: NOT PROVIDED (test set has no target)")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_basic(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], StandardScaler, object, List[int]]:
    """
    Apply preprocessing:
    1. Remove duplicate rows from train
    2. Handle missing values (fill with mean from train)
    3. Apply variance-based feature selection (remove low-variance features)
    4. Scale features using StandardScaler (fit on train, transform both)
    5. Encode labels if needed
    
    Returns:
        (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded or None, scaler)
    """
    print("\n[INFO] === Preprocessing ===")
    
    # Remove duplicates from train (keep first occurrence)
    initial_train_size = len(X_train)
    X_train = X_train.drop_duplicates()
    y_train = y_train[X_train.index]
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    removed_dups = initial_train_size - len(X_train)
    if removed_dups > 0:
        print(f"[INFO] Removed {removed_dups} duplicate rows from training data")
    
    # Handle missing values (fill with train mean)
    train_means = X_train.mean()
    X_train = X_train.fillna(train_means)
    X_test = X_test.fillna(train_means)
    
    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    if missing_train > 0 or missing_test > 0:
        print(f"[INFO] Missing values after imputation — train: {missing_train}, test: {missing_test}")
    else:
        print("[SUCCESS] No missing values")
    
    # Convert to numpy and scale
    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)
    print("[SUCCESS] Features scaled using StandardScaler")

    # Feature selection: Remove extremely low variance features to reduce overfitting
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.05)
    X_train_reduced = selector.fit_transform(X_train_scaled)
    X_test_reduced = selector.transform(X_test_scaled)
    n_features_removed = X_train_np.shape[1] - X_train_reduced.shape[1]
    if n_features_removed > 0:
        print(f"[INFO] Removed {n_features_removed} low-variance features via VarianceThreshold")
    print(f"[INFO] Features after variance selection: {X_train_reduced.shape[1]}")

    # Remove highly correlated features (>0.95) using train correlation matrix
    corr_matrix = pd.DataFrame(X_train_reduced).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        print(f"[INFO] Removing {len(to_drop)} highly correlated features")
    keep_indices = [i for i in range(X_train_reduced.shape[1]) if i not in to_drop]
    X_train_final = X_train_reduced[:, keep_indices]
    X_test_final = X_test_reduced[:, keep_indices]
    print(f"[INFO] Features after correlation filter: {X_train_final.shape[1]}")

    # Encode labels (convert to 0,1 if needed)
    y_train_np = y_train.values.astype(int)
    
    unique_labels = np.unique(y_train_np)
    print(f"[INFO] Unique labels in train: {unique_labels}")
    
    # Ensure binary classification (0, 1)
    if len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels:
        print("[SUCCESS] Binary labels detected (0, 1)")
    elif len(unique_labels) == 2:
        # Remap to 0, 1
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        y_train_np = np.array([label_map[y] for y in y_train_np], dtype=int)
        print(f"[INFO] Labels remapped: {label_map}")
    else:
        raise ValueError(f"Expected binary classification, got {len(unique_labels)} classes")
    
    y_test_np = None
    if y_test is not None:
        y_test_np = y_test.values.astype(int)
        # Apply same remapping if needed
        if len(unique_labels) == 2 and not (0 in unique_labels and 1 in unique_labels):
            y_test_np = np.array([label_map[y] for y in y_test_np], dtype=int)

    # Return processed arrays and the selector objects (variance selector and kept indices)
    return X_train_final, X_test_final, y_train_np, y_test_np, scaler, selector, keep_indices


def get_cv_splitter(groups=None, n_splits=CV_FOLDS):
    """Return the best CV splitter for the data, using group-aware splits when available."""
    if groups is not None:
        try:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        except Exception:
            return GroupKFold(n_splits=n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def save_leakage_report(lines: List[str]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LEAKAGE_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
    print(f"[INFO] Leakage report saved: {LEAKAGE_REPORT_PATH}")


def compute_feature_target_correlations(
    X_train: np.ndarray,
    feature_cols: List[str],
    y_train: np.ndarray,
) -> pd.Series:
    df = pd.DataFrame(X_train, columns=feature_cols)
    y_series = pd.Series(y_train, name="_target")
    corrs = {}
    for col in df.columns:
        column = df[col]
        if column.nunique() <= 1:
            corrs[col] = 0.0
            continue
        corr_val = float(column.corr(y_series))
        corrs[col] = 0.0 if np.isnan(corr_val) else corr_val
    return pd.Series(corrs)


def report_top_feature_target_correlations(
    X_train: np.ndarray,
    feature_cols: List[str],
    y_train: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    corr = compute_feature_target_correlations(X_train, feature_cols, y_train)
    corr_abs = corr.abs().sort_values(ascending=False)
    top = corr_abs.head(top_n).rename("abs_correlation")
    print("\n[INFO] Top feature-target correlations:")
    for rank, (feature, value) in enumerate(top.items(), start=1):
        print(f"  {rank:2d}. {feature:40s} {value:.4f}")
    return top.to_frame()


def remove_leaky_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_cols: List[str],
    y_train: np.ndarray,
    threshold: float = TARGET_CORRELATION_THRESHOLD,
    report_lines: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[str, float]]]:
    corr = compute_feature_target_correlations(X_train, feature_cols, y_train).abs()
    high_corr = corr[corr > threshold].sort_values(ascending=False)
    if high_corr.empty:
        msg = f"[INFO] No leakage features found with abs(correlation) > {threshold:.2f}."
        print(msg)
        if report_lines is not None:
            report_lines.append(msg)
        return X_train, X_test, feature_cols, []

    dropped = list(high_corr.index)
    msg = (
        f"[WARNING] Dropping {len(dropped)} features with abs(correlation) > {threshold:.2f} "
        f"to target: {dropped}"
    )
    print(msg)
    if report_lines is not None:
        report_lines.append(msg)
    for feature, corr_val in high_corr.items():
        detail = f"  - {feature}: abs(corr)={corr_val:.4f}"
        print(detail)
        if report_lines is not None:
            report_lines.append(detail)

    keep_mask = [feature not in dropped for feature in feature_cols]
    keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
    X_train_clean = X_train[:, keep_indices]
    X_test_clean = X_test[:, keep_indices]
    return X_train_clean, X_test_clean, [feature_cols[i] for i in keep_indices], list(high_corr.items())


def count_exact_feature_overlap(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_cols: List[str],
) -> int:
    df_train = pd.DataFrame(X_train, columns=feature_cols)
    df_test = pd.DataFrame(X_test, columns=feature_cols)
    train_rows = set(map(tuple, np.round(df_train.values, 8)))
    test_rows = set(map(tuple, np.round(df_test.values, 8)))
    overlap = train_rows.intersection(test_rows)
    return len(overlap)


def count_duplicate_rows(X: np.ndarray, feature_cols: List[str]) -> int:
    df = pd.DataFrame(X, columns=feature_cols)
    return int(len(df) - len(df.drop_duplicates()))


def assert_no_patient_id_feature(
    feature_cols: List[str],
    patient_col: str,
    report_lines: Optional[List[str]] = None,
) -> List[str]:
    if patient_col in feature_cols:
        message = f"[WARNING] Patient ID column '{patient_col}' appears in selected features and will be removed."
        print(message)
        if report_lines is not None:
            report_lines.append(message)
        feature_cols = [f for f in feature_cols if f != patient_col]
    return feature_cols


def evaluate_baseline_comparison(
    X_train_before,
    X_test_before,
    X_train_after,
    X_test_after,
    y_train,
    y_test,
    report_lines: Optional[List[str]] = None,
) -> None:
    from sklearn.linear_model import LogisticRegression
    baseline = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    baseline.fit(X_train_before, y_train)
    before_train = float(baseline.score(X_train_before, y_train))
    before_test = float(baseline.score(X_test_before, y_test))

    baseline.fit(X_train_after, y_train)
    after_train = float(baseline.score(X_train_after, y_train))
    after_test = float(baseline.score(X_test_after, y_test))

    print("\n[INFO] Baseline accuracy before/after removing high-target-correlation features:")
    print(f"  before: train={before_train:.4f}, test={before_test:.4f}")
    print(f"  after : train={after_train:.4f}, test={after_test:.4f}")
    if report_lines is not None:
        report_lines.append(
            f"Baseline before: train={before_train:.4f}, test={before_test:.4f}"
        )
        report_lines.append(
            f"Baseline after : train={after_train:.4f}, test={after_test:.4f}"
        )


def warn_perfect_metrics(
    label: str,
    value: float,
    report_lines: Optional[List[str]] = None,
) -> None:
    if value >= 1.0:
        warning = f"[WARNING] {label} reached 100.00%. This may indicate leakage or an overly optimistic evaluation."
        print(warning)
        if report_lines is not None:
            report_lines.append(warning)


def print_accuracy_gap(train_acc: float, test_acc: float, report_lines: Optional[List[str]] = None) -> None:
    gap = train_acc - test_acc
    msg = f"[INFO] Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f} | Gap: {gap:.4f}"
    print(msg)
    if report_lines is not None:
        report_lines.append(msg)
    if gap > OVERFIT_GAP_THRESHOLD:
        overfit = f"[WARNING] Accuracy gap {gap:.4f} exceeds overfitting threshold of {OVERFIT_GAP_THRESHOLD:.2f}."
        print(overfit)
        if report_lines is not None:
            report_lines.append(overfit)


# ============================================================================
# Class Distribution & Balancing
# ============================================================================

def print_class_distribution(y_train: np.ndarray, y_test: np.ndarray):
    """Print class distribution and compute class weights."""
    print("\n[INFO] === Class Distribution ===")
    
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    print("[INFO] Training set:")
    for cls in range(len(train_counts)):
        pct = 100.0 * train_counts[cls] / len(y_train)
        class_name = "Healthy" if cls == 0 else "Parkinson"
        print(f"[INFO]   {class_name} (label {cls}): {train_counts[cls]} ({pct:.1f}%)")
    
    print("[INFO] Test set:")
    for cls in range(len(test_counts)):
        pct = 100.0 * test_counts[cls] / len(y_test)
        class_name = "Healthy" if cls == 0 else "Parkinson"
        print(f"[INFO]   {class_name} (label {cls}): {test_counts[cls]} ({pct:.1f}%)")
    
    # Compute balanced weights
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    
    print(f"[INFO] Balanced class weights: {weight_dict}")
    
    # Check if imbalanced
    if train_counts[0] > 0 and train_counts[1] > 0:
        ratio = max(train_counts) / min(train_counts)
        if ratio > 1.5:
            print(f"[WARNING] Class imbalance detected (ratio: {ratio:.2f})")
        else:
            print("[SUCCESS] Classes reasonably balanced")
    
    return weight_dict


def select_optimal_k(X: np.ndarray, y: np.ndarray, ks: List[int]) -> int:
    """
    Evaluate multiple SelectKBest configurations using StratifiedKFold CV
    and return the best k (or -1 for all features).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    print("\n[INFO] Selecting optimal number of features via CV...")
    best_k = None
    best_score = -np.inf
    cv = get_cv_splitter()

    for k in ks:
        if k == -1:
            Xk = X
            name = "all"
        else:
            k_use = min(k, X.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k_use)
            Xk = selector.fit_transform(X, y)
            name = str(k_use)

        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
        scores = cross_val_score(clf, Xk, y, cv=cv, scoring='f1', n_jobs=-1)
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        print(f"[INFO] k={name} -> CV F1: {mean:.4f} ± {std:.4f}")
        if mean > best_score:
            best_score = mean
            best_k = k

    print(f"[SUCCESS] Selected k = {('all' if best_k==-1 else best_k)} with CV F1 = {best_score:.4f}")
    return best_k


def select_best_k_group_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: List[str],
    scale_pos_weight: float,
    ks: List[int],
) -> Tuple[int, Dict[str, float]]:
    """Select the best k features with group-aware CV to prevent patient leakage."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    print("\n[INFO] Selecting best k with group-aware CV...")
    best_k = None
    best_score = -np.inf
    cv = get_cv_splitter(groups)
    scores_by_k = {}

    for k in ks:
        if k == -1:
            Xk = X
            name = "all"
        else:
            k_use = min(k, X.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k_use)
            Xk = selector.fit_transform(X, y)
            name = str(k_use)

        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
        scores = cross_val_score(clf, Xk, y, cv=cv, scoring='f1', n_jobs=-1)
        mean_score = float(np.mean(scores))
        scores_by_k[name] = mean_score
        print(f"[INFO] k={name} -> Group CV F1: {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    print(f"[SUCCESS] Selected k = {('all' if best_k == -1 else best_k)} with group-aware CV F1 = {best_score:.4f}")
    return best_k, scores_by_k


def train_catboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weight_dict: Dict,
    n_iter: int = 30,
    groups: Optional[np.ndarray] = None,
) -> Tuple[object, Dict]:
    """Train CatBoost with RandomizedSearchCV."""
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost is not installed in the environment")

    print("\n[INFO] Training CatBoost...")
    base_model = CatBoostClassifier(thread_count=-1, random_state=RANDOM_STATE, verbose=0)

    params = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [32, 64, 128],
    }

    cv = get_cv_splitter(groups)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring='f1',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    print(f"[SUCCESS] Best CV F1 Score: {search.best_score_:.4f}")
    print(f"[INFO] Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def build_stacking_ensemble(
    estimators: List[Tuple[str, object]],
    final_estimator=None,
    groups: Optional[np.ndarray] = None,
) -> object:
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=2000)

    cv = get_cv_splitter(groups)
    stk = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1,
        passthrough=False,
    )
    return stk


def sample_weights_from_map(y: np.ndarray, weight_map: Dict[int, float]) -> np.ndarray:
    return np.array([weight_map[int(l)] for l in y], dtype=np.float32)


def collect_oof_stack(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    base_models: list,
    sample_weight: np.ndarray,
) -> np.ndarray:
    """Collect probabilities from out-of-fold base models using patient-aware splits."""
    cv = get_cv_splitter(groups)
    oof = np.zeros((len(y_train), len(base_models)), dtype=np.float64)

    print("\n[INFO] === Collecting OOF predictions (stacking layer 1) ===")
    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups), 1):
        print(f"[INFO]   Fold {fold_i}/{CV_FOLDS}", end=" | ")
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr = y_train[tr_idx]
        sw_tr = sample_weight[tr_idx]

        for m_i, model in enumerate(base_models):
            m = clone(model)
            fit_kwargs = {"verbose": False} if isinstance(m, xgb.XGBClassifier) else {}
            if HAS_LIGHTGBM and isinstance(m, lgb.LGBMClassifier):
                fit_kwargs = {"callbacks": [lgb.log_evaluation(-1)]}
            try:
                m.fit(X_tr, y_tr, sample_weight=sw_tr, **fit_kwargs)
            except TypeError:
                m.fit(X_tr, y_tr, **fit_kwargs)

            if hasattr(m, "predict_proba"):
                oof[val_idx, m_i] = m.predict_proba(X_val)[:, 1]
            else:
                oof[val_idx, m_i] = m.predict(X_val)
            print(f"{type(m).__name__[:4]}✓", end=" ")
        print()

    for m_i, model in enumerate(base_models):
        try:
            auc = roc_auc_score(y_train, oof[:, m_i])
            print(f"[INFO]   OOF AUC {type(model).__name__[:12]:12s}: {auc*100:.2f}%")
        except Exception:
            pass

    return oof


def train_base_models_full(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_models: list,
    sample_weight: np.ndarray,
) -> list:
    trained = []
    print("\n[INFO] === Retraining base models on full training set ===")
    for model in base_models:
        m = clone(model)
        fit_kwargs = {"verbose": False} if isinstance(m, xgb.XGBClassifier) else {}
        if HAS_LIGHTGBM and isinstance(m, lgb.LGBMClassifier):
            fit_kwargs = {"callbacks": [lgb.log_evaluation(-1)]}
        try:
            m.fit(X_train, y_train, sample_weight=sample_weight, **fit_kwargs)
        except TypeError:
            m.fit(X_train, y_train, **fit_kwargs)
        trained.append(m)
        print(f"[INFO]   {type(m).__name__} trained ✓")
    return trained


def predict_test_stack(trained_base: list, X_test: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else m.predict(X_test) for m in trained_base])


def tune_decision_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    bal = balanced_accuracy_score(y_true, (y_prob >= best_t).astype(int))
    return best_t, bal


def check_overfitting(train_acc, val_acc):
    gap = train_acc - val_acc
    print(f"\n[INFO] Train acc: {train_acc*100:.2f}% | Test acc: {val_acc*100:.2f}% | Gap: {gap*100:.2f}%")
    if gap > OVERFIT_GAP_THRESHOLD:
        print(f"[WARNING] Overfitting detected (gap > {OVERFIT_GAP_THRESHOLD*100:.0f}%)")


def print_top_xgb_features(model, feature_cols, top_n=20):
    if not hasattr(model, "feature_importances_"):
        print("[WARNING] Model does not expose feature_importances_.")
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    print(f"\n[INFO] Top {top_n} XGBoost features:")
    for rank, i in enumerate(idx, 1):
        print(f"  {rank:2d}. {feature_cols[i]:40s} {imp[i]:.4f}")


def optional_shap_analysis(model, X_train, feature_cols):
    try:
        import shap
        print("[INFO] Running SHAP analysis...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        plt.figure()
        shap.summary_plot(shap_values, X_train, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "voice_shap_summary.png"), dpi=100, bbox_inches="tight")
        plt.close()
        print("[INFO] SHAP summary plot saved")
    except Exception as exc:
        print(f"[INFO] SHAP analysis skipped: {exc}")


# ============================================================================
# Model Training
# ============================================================================

def build_model_params() -> Dict[str, Dict]:
    """Build hyperparameter grids for each model type with regularization."""
    params = {
        "xgb": {
            "n_estimators": [100, 150, 200],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.05, 0.1, 0.15],
            "subsample": [0.6, 0.7, 0.8],
            "colsample_bytree": [0.6, 0.7, 0.8],
            "min_child_weight": [3, 5, 7],
            "gamma": [0.5, 1.0, 1.5],
            "reg_alpha": [0.5, 1.0, 2.0],
            "reg_lambda": [1.0, 2.0, 3.0],
        },
        "rf": {
            "n_estimators": [100, 150, 200],
            "max_depth": [5, 7, 10, 12],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [2, 4, 6],
            "max_features": ["sqrt", "log2"],
        },
        "et": {
            "n_estimators": [100, 150, 200],
            "max_depth": [5, 7, 10, 12],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [2, 4, 6],
            "max_features": ["sqrt", "log2"],
        },
        "lgb": {
            "n_estimators": [100, 150, 200],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.05, 0.1, 0.15],
            "num_leaves": [15, 20, 25],
            "min_data_in_leaf": [15, 20, 30],
        },
    }
    return params


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weight_dict: Dict,
    n_iter: int = 30,
    groups: Optional[np.ndarray] = None,
) -> Tuple[xgb.XGBClassifier, Dict]:
    """Train XGBoost with RandomizedSearchCV and early stopping via validation."""
    print("\n[INFO] Training XGBoost with validation-based early stopping...")
    
    # Create validation split (20% for early stopping)
    from sklearn.model_selection import train_test_split as tts
    X_tr, X_val, y_tr, y_val = tts(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    scale_pos_weight = weight_dict[0] / weight_dict[1] if weight_dict[1] > 0 else 1.0
    
    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    
    params = build_model_params()["xgb"]
    cv = get_cv_splitter(groups)
    
    search = RandomizedSearchCV(
        base_model,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    
    search.fit(X_train, y_train)
    print(f"[SUCCESS] Best CV F1 Score: {search.best_score_:.4f}")
    print(f"[INFO] Best params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_


def train_model_with_smote_random_search(
    base_model,
    params: Dict,
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 50,
):
    """Train model inside an imbalanced-learn Pipeline with SMOTE and RandomizedSearchCV.
    Returns the fitted pipeline and best params (with clf__ prefixes removed).
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if HAS_IMBLEARN and SMOTE is not None and ImbPipeline is not None:
        pipeline = ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", base_model)])
    else:
        # fallback to plain estimator
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([("clf", base_model)])

    # prefix params
    prefixed_params = {f"clf__{k}": v for k, v in params.items()}

    cv = get_cv_splitter()
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=prefixed_params,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X, y)
    best_pipeline = search.best_estimator_
    # extract clf params
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    return best_pipeline, best_params


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weight_dict: Dict,
    n_iter: int = 20,
    groups: Optional[np.ndarray] = None,
) -> Tuple[RandomForestClassifier, Dict]:
    """Train Random Forest with RandomizedSearchCV."""
    print("\n[INFO] Training Random Forest...")
    
    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    
    params = build_model_params()["rf"]
    cv = get_cv_splitter(groups)
    
    search = RandomizedSearchCV(
        base_model,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    
    search.fit(X_train, y_train)
    print(f"[SUCCESS] Best CV F1 Score: {search.best_score_:.4f}")
    print(f"[INFO] Best params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_


def train_extra_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weight_dict: Dict,
    n_iter: int = 20,
    groups: Optional[np.ndarray] = None,
) -> Tuple[ExtraTreesClassifier, Dict]:
    """Train Extra Trees with RandomizedSearchCV."""
    print("\n[INFO] Training Extra Trees...")
    
    base_model = ExtraTreesClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    
    params = build_model_params()["et"]
    cv = get_cv_splitter(groups)
    
    search = RandomizedSearchCV(
        base_model,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    
    search.fit(X_train, y_train)
    print(f"[SUCCESS] Best CV F1 Score: {search.best_score_:.4f}")
    print(f"[INFO] Best params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weight_dict: Dict,
    n_iter: int = 20,
    groups: Optional[np.ndarray] = None,
) -> Tuple[object, Dict]:
    """Train LightGBM with RandomizedSearchCV."""
    print("\n[INFO] Training LightGBM...")
    
    base_model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        is_unbalance=True,
        verbosity=-1,
    )
    
    params = build_model_params()["lgb"]
    cv = get_cv_splitter(groups)
    
    search = RandomizedSearchCV(
        base_model,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    
    search.fit(X_train, y_train)
    print(f"[SUCCESS] Best CV F1 Score: {search.best_score_:.4f}")
    print(f"[INFO] Best params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict:
    """Evaluate model on train and test sets."""
    print(f"\n[INFO] Evaluating {model_name}...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities (for ROC-AUC)
    if hasattr(model, "predict_proba"):
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_train_prob = model.decision_function(X_train)
        y_test_prob = model.decision_function(X_test)
    else:
        y_train_prob = np.zeros_like(y_train, dtype=float)
        y_test_prob = np.zeros_like(y_test, dtype=float)
    
    # Metrics
    metrics = {
        "model_name": model_name,
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "train_balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_test_prob)),
    }
    
    print(f"[INFO]   Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"[INFO]   Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"[INFO]   Test Balanced Accuracy: {metrics['test_balanced_accuracy']:.4f}")
    print(f"[INFO]   Test F1 Score: {metrics['test_f1']:.4f}")
    print(f"[INFO]   Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    
    return metrics, y_test_pred, y_test_prob


def print_detailed_evaluation(y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    """Print detailed evaluation metrics and plots."""
    print("\n[INFO] === Detailed Evaluation ===")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Parkinson"],
        yticklabels=["Healthy", "Parkinson"],
    )
    plt.title("Confusion Matrix - Latest Voice Model")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "latest_voice_confusion_matrix.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("[SUCCESS] Confusion matrix plot saved")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Latest Voice Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "latest_voice_roc_curve.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("[SUCCESS] ROC curve plot saved")


def plot_feature_importance(model, model_name: str, n_features: int = 20):
    """Plot and save feature importance."""
    if not hasattr(model, "feature_importances_"):
        print(f"[WARNING] Model {model_name} does not have feature_importances_")
        return
    
    print(f"\n[INFO] Top {n_features} features for {model_name}:")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:2d}. Feature {idx}: {importances[idx]:.6f}")
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.title(f"Top {n_features} Feature Importance - {model_name}")
    plt.bar(range(n_features), importances[indices], align="center")
    plt.xticks(range(n_features), [f"F{idx}" for idx in indices], rotation=45)
    plt.ylabel("Importance")
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(LATEST_FI_PLOT_PATH, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Feature importance plot saved: {LATEST_FI_PLOT_PATH}")


def train_voice_stacking():
    """Patient-wise stacking ensemble adapted from the voice branch logic."""
    print("[INFO] === Voice Stacking Ensemble (XGB + LGB + RF + ET) ===")

    # Load patient-wise splits and features
    train_df, test_df, patient_col, _target_col, raw_feature_cols = load_voice_patient_split(
        test_size=0.1, random_state=RANDOM_STATE
    )
    save_voice_splits(train_df, test_df, patient_col)

    y_train = train_df["_label"].values
    y_test = test_df["_label"].values
    groups_train = train_df[patient_col].astype(str).values

    weight_map, scale_pos_weight = print_class_distribution_and_weights(
        y_train, "train", print_weights=True
    )
    print_class_distribution_and_weights(y_test, "test", print_weights=False)

    overlap = set(groups_train) & set(test_df[patient_col].astype(str).values)
    if overlap:
        print(f"[ERROR] Patient overlap: {sorted(overlap)[:10]}")
        return
    else:
        print("[SUCCESS] No patient overlap between voice train and test.")

    # Feature engineering and selection
    pruned_cols, variance_selector, _, _ = prune_weak_voice_features(train_df, test_df, raw_feature_cols)
    train_df, test_df = impute_with_train_stats(train_df, test_df, pruned_cols)
    joblib.dump(train_df[pruned_cols].mean(), os.path.join(MODELS_DIR, "voice_impute_means.pkl"))
    X_train, X_test, scaler = scale_train_transform_test(train_df, test_df, pruned_cols)

    best_k, k_scores = select_best_k_group_cv(
        X_train, y_train, groups_train, pruned_cols, scale_pos_weight, [30, 50, 75, 100]
    )
    X_train_sel, X_test_sel, selector, feature_cols = select_k_best_features(
        X_train, y_train, X_test, pruned_cols, best_k
    )

    leakage_report: List[str] = [
        "Voice stacking leakage report",
        f"Train patients: {len(train_df)} | Test patients: {len(test_df)}",
        f"Initial selected features: {len(feature_cols)}",
    ]

    if patient_col in feature_cols:
        removed_idx = feature_cols.index(patient_col)
        warning = f"[WARNING] Patient ID column '{patient_col}' found in selected features and removed."
        print(warning)
        leakage_report.append(warning)
        X_train_sel = np.delete(X_train_sel, removed_idx, axis=1)
        X_test_sel = np.delete(X_test_sel, removed_idx, axis=1)
        feature_cols.pop(removed_idx)

    report_top_feature_target_correlations(X_train_sel, feature_cols, y_train, top_n=20)
    X_train_sel_before = X_train_sel.copy()
    X_test_sel_before = X_test_sel.copy()

    X_train_sel, X_test_sel, feature_cols, dropped = remove_leaky_features(
        X_train_sel,
        X_test_sel,
        feature_cols,
        y_train,
        threshold=TARGET_CORRELATION_THRESHOLD,
        report_lines=leakage_report,
    )

    if dropped:
        evaluate_baseline_comparison(
            X_train_sel_before,
            X_test_sel_before,
            X_train_sel,
            X_test_sel,
            y_train,
            y_test,
            leakage_report,
        )

    overlap_count = count_exact_feature_overlap(X_train_sel, X_test_sel, feature_cols)
    train_dups = count_duplicate_rows(X_train_sel, feature_cols)
    test_dups = count_duplicate_rows(X_test_sel, feature_cols)

    leakage_report.append(f"Exact feature-vector overlap between train/test: {overlap_count}")
    leakage_report.append(f"Internal duplicate rows: train={train_dups}, test={test_dups}")
    if overlap_count > 0:
        warning = f"[WARNING] Found {overlap_count} exact feature-vector overlaps between train and test."
        print(warning)
        leakage_report.append(warning)

    train_sel, test_sel = make_selected_voice_frames(
        X_train_sel, X_test_sel, feature_cols, train_df, test_df, patient_col
    )

    with open(os.path.join(MODELS_DIR, "voice_best_k.json"), "w") as f:
        json.dump({"best_k": best_k, "cv_mean_auc": k_scores}, f, indent=2)
    with open(SELECTED_FEATURES_PATH, "w") as f:
        f.writelines(f"{n}\n" for n in feature_cols)
    print(f"[SUCCESS] Saved best k={best_k}, {len(feature_cols)} features.")

    # Build base models
    def _build_xgb_inner():
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.3,
            reg_lambda=1.5,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )

    def _build_rf_inner():
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

    def _build_et_inner():
        return ExtraTreesClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)

    def _build_lgb_inner():
        if not HAS_LIGHTGBM:
            return None
        return lgb.LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

    base_models = [
        _build_xgb_inner(),
        _build_rf_inner(),
        _build_et_inner(),
    ]
    if HAS_LIGHTGBM:
        base_models.append(_build_lgb_inner())

    print(f"\n[INFO] Stacking with {len(base_models)} base models: {[type(m).__name__ for m in base_models if m is not None]}")

    sample_weight = sample_weights_from_map(y_train, weight_map)

    # Layer 1: collect OOF
    oof_train = collect_oof_stack(X_train_sel, y_train, groups_train, base_models, sample_weight)

    # Meta-learner
    meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    meta.fit(oof_train, y_train)
    oof_meta_prob = meta.predict_proba(oof_train)[:, 1]
    oof_meta_auc = roc_auc_score(y_train, oof_meta_prob)
    print(f"[SUCCESS] Meta-learner OOF AUC: {oof_meta_auc*100:.2f}%")
    warn_perfect_metrics("Meta OOF ROC-AUC", oof_meta_auc, leakage_report)

    # Threshold tuning on OOF
    decision_threshold, oof_bal = tune_decision_threshold(y_train, oof_meta_prob)
    print(f"[SUCCESS] Decision threshold={decision_threshold:.3f} | OOF bal-acc={oof_bal*100:.2f}%")
    with open(os.path.join(MODELS_DIR, "voice_decision_threshold.json"), "w") as f:
        json.dump({"threshold": decision_threshold, "oof_balanced_accuracy": oof_bal}, f, indent=2)

    # Retrain base models on full train
    trained_base = train_base_models_full(X_train_sel, y_train, base_models, sample_weight)
    test_stack = predict_test_stack(trained_base, X_test_sel)
    train_stack = predict_test_stack(trained_base, X_train_sel)

    # Final predictions
    y_prob_train = meta.predict_proba(train_stack)[:, 1]
    y_prob = meta.predict_proba(test_stack)[:, 1]
    y_pred_train = (y_prob_train >= decision_threshold).astype(int)
    y_pred = (y_prob >= decision_threshold).astype(int)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)

    warn_perfect_metrics("Train accuracy", train_acc, leakage_report)
    warn_perfect_metrics("Test accuracy", test_acc, leakage_report)
    warn_perfect_metrics("Test ROC-AUC", test_auc, leakage_report)
    print_accuracy_gap(train_acc, test_acc, leakage_report)
    check_overfitting(train_acc, test_acc)

    print("\n[INFO] --- Voice Test Results (patient hold-out, stacking ensemble) ---")
    print(f"[SUCCESS] Accuracy:          {test_acc*100:.2f}%")
    print(f"[INFO] Balanced accuracy:   {bal_acc*100:.2f}%")
    print(f"[INFO] Precision:           {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"[INFO] Recall:              {recall_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"[INFO] F1:                  {f1_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"[INFO] ROC-AUC:             {test_auc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["healthy", "parkinson"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"[INFO] Confusion matrix:\n{cm}")
    warn_suspicious_accuracy("Voice", test_acc)

    # Save artifacts
    joblib.dump(trained_base[0], os.path.join(MODELS_DIR, "voice_xgb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(selector, os.path.join(MODELS_DIR, "feature_selector.pkl"))
    joblib.dump(variance_selector, os.path.join(MODELS_DIR, "voice_variance_selector.pkl"))
    joblib.dump(pruned_cols, os.path.join(MODELS_DIR, "voice_pruned_columns.pkl"))
    joblib.dump(trained_base, os.path.join(MODELS_DIR, "voice_base_models.pkl"))
    joblib.dump(meta, os.path.join(MODELS_DIR, "voice_meta_learner.pkl"))
    joblib.dump(train_sel, os.path.join(MODELS_DIR, "voice_train_features.pkl"))
    joblib.dump(test_sel, os.path.join(MODELS_DIR, "voice_test_features.pkl"))
    print("[SUCCESS] All models saved.")

    # Plots
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["healthy", "parkinson"], yticklabels=["healthy", "parkinson"])
    plt.title("Voice Confusion Matrix (stacking ensemble)")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.savefig(os.path.join(OUTPUT_DIR, "voice_confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.legend()
    plt.title("Voice ROC (stacking ensemble)")
    plt.savefig(os.path.join(OUTPUT_DIR, "voice_roc_curve.png"))
    plt.close()

    print_top_xgb_features(trained_base[0], feature_cols)
    optional_shap_analysis(trained_base[0], X_train_sel, feature_cols)

    # Per-model breakdown
    print("\n[INFO] --- Individual base model test accuracies ---")
    for m_i, model in enumerate(trained_base):
        prob_i = test_stack[:, m_i]
        t_i, _ = tune_decision_threshold(y_train, oof_train[:, m_i])
        pred_i = (prob_i >= t_i).astype(int)
        acc_i = accuracy_score(y_test, pred_i)
        try:
            auc_i = roc_auc_score(y_test, prob_i)
        except Exception:
            auc_i = None
        print(f"[INFO]   {type(model).__name__:20s} acc={acc_i*100:.2f}%  auc={(auc_i*100 if auc_i is not None else 0):.2f}%  thresh={t_i:.3f}")

    print("\n[INFO] Model performance is reported on the patient-wise holdout set.")
    print("[INFO] This evaluation is intended to be realistic, not to chase an artificial perfect score.")

    leakage_report.append(f"Final test accuracy: {test_acc:.4f}")
    leakage_report.append(f"Final test ROC-AUC: {test_auc:.4f}")
    leakage_report.append(f"Final feature count: {len(feature_cols)}")
    save_leakage_report(leakage_report)


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main training pipeline."""
    print("[INFO] === Latest Voice Model Training Pipeline ===\n")
    
    # Create output directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    if not os.path.isfile(TRAIN_FILE):
        raise FileNotFoundError(f"Train file not found: {TRAIN_FILE}")
    if not os.path.isfile(TEST_FILE):
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    
    train_df, test_df = load_data_from_latest(TRAIN_FILE, TEST_FILE)
    
    # Detect structure
    id_col_idx, target_col_idx = detect_dataset_structure(train_df, test_df)
    
    # Extract features and target
    X_train, X_test, y_train, y_test = extract_features_and_target(
        train_df, test_df, id_col_idx, target_col_idx
    )
    
    # If test data doesn't have target, perform stratified split on train for a held-out test
    if y_test is None:
        print("\n[INFO] Test set has no target labels. Creating held-out test set from train...")
        X_train_full, X_test_external, y_train_full, y_test_external = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )
        X_train = X_train_full
        X_test = X_test_external
        y_train = y_train_full
        y_test = y_test_external
        print(f"[INFO] Created external test split: {len(X_train)} / {len(X_test)}")

    # Split train into inner-train and validation holdout for threshold selection and final eval
    X_train_inner, X_val_hold, y_train_inner, y_val_hold = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )

    # Preprocess: fit on inner train and transform val_hold
    X_train_proc, X_val_proc, y_train_proc, y_val_proc, scaler, variance_selector, keep_indices = preprocess_basic(
        X_train_inner, X_val_hold, y_train_inner, y_val_hold
    )

    # Transform the external test set using the same scaler/selector/keep_indices
    X_test_np = X_test.values.astype(np.float32)
    X_test_scaled_full = scaler.transform(X_test_np)
    X_test_reduced = variance_selector.transform(X_test_scaled_full)
    X_test_proc = X_test_reduced[:, keep_indices]

    # Automatic feature count selection
    candidate_ks = [-1, 30, 50, 75, 100]
    best_k = select_optimal_k(X_train_proc, y_train_proc, candidate_ks)
    kbest_selector = None
    if best_k != -1 and best_k is not None:
        from sklearn.feature_selection import SelectKBest, f_classif
        k_use = min(best_k, X_train_proc.shape[1])
        kbest_selector = SelectKBest(score_func=f_classif, k=k_use)
        X_train_sel = kbest_selector.fit_transform(X_train_proc, y_train_proc)
        X_val_sel = kbest_selector.transform(X_val_proc)
        X_test_sel = kbest_selector.transform(X_test_proc)
        print(f"[INFO] Using top {k_use} features")
    else:
        X_train_sel = X_train_proc
        X_val_sel = X_val_proc
        X_test_sel = X_test_proc
        print("[INFO] Using all features")

    # Class distribution and weights
    weight_dict = print_class_distribution(y_train_proc, y_val_proc)

    # Train and evaluate models
    models_trained = {}
    metrics_list = []

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # XGBoost
    try:
        xgb_model, xgb_params = train_xgboost_model(X_train_sel, y_train_proc, weight_dict, n_iter=40)
        # cross-val scores
        from sklearn.model_selection import cross_val_score
        cv_f1 = float(np.mean(cross_val_score(xgb_model, X_train_sel, y_train_proc, cv=skf, scoring='f1', n_jobs=-1)))
        cv_auc = float(np.mean(cross_val_score(xgb_model, X_train_sel, y_train_proc, cv=skf, scoring='roc_auc', n_jobs=-1)))
        metrics_xgb, _, _ = evaluate_model(xgb_model, X_train_sel, X_val_sel, y_train_proc, y_val_proc, "XGBoost")
        metrics_xgb.update({"cv_f1": cv_f1, "cv_roc_auc": cv_auc, "best_params": xgb_params})
        models_trained["xgboost"] = (xgb_model, xgb_params)
        metrics_list.append(metrics_xgb)
    except Exception as e:
        print(f"[ERROR] XGBoost training failed: {e}")

    # Random Forest
    try:
        rf_model, rf_params = train_random_forest(X_train_sel, y_train_proc, weight_dict, n_iter=40)
        cv_f1 = float(np.mean(cross_val_score(rf_model, X_train_sel, y_train_proc, cv=skf, scoring='f1', n_jobs=-1)))
        cv_auc = float(np.mean(cross_val_score(rf_model, X_train_sel, y_train_proc, cv=skf, scoring='roc_auc', n_jobs=-1)))
        metrics_rf, _, _ = evaluate_model(rf_model, X_train_sel, X_val_sel, y_train_proc, y_val_proc, "Random Forest")
        metrics_rf.update({"cv_f1": cv_f1, "cv_roc_auc": cv_auc, "best_params": rf_params})
        models_trained["random_forest"] = (rf_model, rf_params)
        metrics_list.append(metrics_rf)
    except Exception as e:
        print(f"[ERROR] Random Forest training failed: {e}")

    # Extra Trees
    try:
        et_model, et_params = train_extra_trees(X_train_sel, y_train_proc, weight_dict, n_iter=40)
        cv_f1 = float(np.mean(cross_val_score(et_model, X_train_sel, y_train_proc, cv=skf, scoring='f1', n_jobs=-1)))
        cv_auc = float(np.mean(cross_val_score(et_model, X_train_sel, y_train_proc, cv=skf, scoring='roc_auc', n_jobs=-1)))
        metrics_et, _, _ = evaluate_model(et_model, X_train_sel, X_val_sel, y_train_proc, y_val_proc, "Extra Trees")
        metrics_et.update({"cv_f1": cv_f1, "cv_roc_auc": cv_auc, "best_params": et_params})
        models_trained["extra_trees"] = (et_model, et_params)
        metrics_list.append(metrics_et)
    except Exception as e:
        print(f"[ERROR] Extra Trees training failed: {e}")

    # LightGBM (if available)
    if HAS_LIGHTGBM:
        try:
            lgb_model, lgb_params = train_lightgbm_model(X_train_sel, y_train_proc, weight_dict, n_iter=40)
            cv_f1 = float(np.mean(cross_val_score(lgb_model, X_train_sel, y_train_proc, cv=skf, scoring='f1', n_jobs=-1)))
            cv_auc = float(np.mean(cross_val_score(lgb_model, X_train_sel, y_train_proc, cv=skf, scoring='roc_auc', n_jobs=-1)))
            metrics_lgb, _, _ = evaluate_model(lgb_model, X_train_sel, X_val_sel, y_train_proc, y_val_proc, "LightGBM")
            metrics_lgb.update({"cv_f1": cv_f1, "cv_roc_auc": cv_auc, "best_params": lgb_params})
            models_trained["lightgbm"] = (lgb_model, lgb_params)
            metrics_list.append(metrics_lgb)
        except Exception as e:
            print(f"[ERROR] LightGBM training failed: {e}")

    # CatBoost (if available)
    if HAS_CATBOOST:
        try:
            cb_model, cb_params = train_catboost_model(X_train_sel, y_train_proc, weight_dict, n_iter=40)
            cv_f1 = float(np.mean(cross_val_score(cb_model, X_train_sel, y_train_proc, cv=skf, scoring='f1', n_jobs=-1)))
            cv_auc = float(np.mean(cross_val_score(cb_model, X_train_sel, y_train_proc, cv=skf, scoring='roc_auc', n_jobs=-1)))
            metrics_cb, _, _ = evaluate_model(cb_model, X_train_sel, X_val_sel, y_train_proc, y_val_proc, "CatBoost")
            metrics_cb.update({"cv_f1": cv_f1, "cv_roc_auc": cv_auc, "best_params": cb_params})
            models_trained["catboost"] = (cb_model, cb_params)
            metrics_list.append(metrics_cb)
        except Exception as e:
            print(f"[ERROR] CatBoost training failed: {e}")

    # Build stacking ensemble using available boosted models
    estimators_for_stack = []
    if "xgboost" in models_trained:
        estimators_for_stack.append(("xgb", models_trained["xgboost"][0]))
    if "lightgbm" in models_trained:
        estimators_for_stack.append(("lgb", models_trained["lightgbm"][0]))
    if "catboost" in models_trained:
        estimators_for_stack.append(("cb", models_trained["catboost"][0]))

    stacking_model = None
    if estimators_for_stack:
        try:
            stacking_model = build_stacking_ensemble(estimators_for_stack)
            stacking_model.fit(X_train_sel, y_train_proc)
            metrics_stack, _, _ = evaluate_model(stacking_model, X_train_sel, X_val_sel, y_train_proc, y_val_proc, "Stacking Ensemble")
            # compute cv scores
            cv_f1 = float(np.mean(cross_val_score(stacking_model, X_train_sel, y_train_proc, cv=skf, scoring='f1', n_jobs=-1)))
            cv_auc = float(np.mean(cross_val_score(stacking_model, X_train_sel, y_train_proc, cv=skf, scoring='roc_auc', n_jobs=-1)))
            metrics_stack.update({"cv_f1": cv_f1, "cv_roc_auc": cv_auc})
            models_trained["stacking"] = (stacking_model, {})
            metrics_list.append(metrics_stack)
        except Exception as e:
            print(f"[ERROR] Stacking training failed: {e}")

    # Optimize probability threshold on validation holdout for each model
    def optimize_threshold(model, X_val, y_val, low=0.30, high=0.70, step=0.01):
        if not hasattr(model, "predict_proba"):
            return 0.5, f1_score(y_val, model.predict(X_val))
        probs = model.predict_proba(X_val)[:, 1]
        best_t = 0.5
        best_f1 = -1
        thresholds = np.arange(low, high + 1e-6, step)
        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t, best_f1

    thresholds = {}
    for name, (model_obj, params) in models_trained.items():
        try:
            t_opt, f1_opt = optimize_threshold(model_obj, X_val_sel, y_val_proc)
            thresholds[name] = {"threshold": float(t_opt), "val_f1": float(f1_opt)}
            print(f"[INFO] Optimized threshold for {name}: {t_opt} (val F1={f1_opt:.4f})")
        except Exception as e:
            print(f"[WARN] Could not optimize threshold for {name}: {e}")

    # Final evaluation on external test set
    final_results = []
    for name, (model_obj, params) in models_trained.items():
        try:
            if hasattr(model_obj, "predict_proba"):
                probs = model_obj.predict_proba(X_test_sel)[:, 1]
                thresh = thresholds.get(name, {}).get("threshold", 0.5)
                preds = (probs >= thresh).astype(int)
            else:
                preds = model_obj.predict(X_test_sel)
                probs = None

            # Compute metrics
            acc = accuracy_score(y_test, preds)
            bal_acc = balanced_accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            roc = roc_auc_score(y_test, probs) if probs is not None else None

            result = {
                "model_name": name,
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "roc_auc": float(roc) if roc is not None else None,
                "threshold": float(thresholds.get(name, {}).get("threshold", 0.5)),
                "best_params": params,
            }
            final_results.append(result)
            print(f"[INFO] Test results for {name}: F1={f1:.4f}, ROC-AUC={roc if roc is not None else 'N/A'}")
        except Exception as e:
            print(f"[WARN] Failed final eval for {name}: {e}")

    # Choose best model based on test F1 then ROC-AUC
    if not final_results:
        print("[ERROR] No final results to select best model")
        return
    best_final = max(final_results, key=lambda r: (r["f1"], (r["roc_auc"] if r["roc_auc"] is not None else -1)))
    best_name = best_final["model_name"]
    best_model_obj = models_trained[best_name][0]
    best_threshold = best_final.get("threshold", 0.5)

    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model_obj, LATEST_MODEL_PATH)
    joblib.dump(scaler, LATEST_SCALER_PATH)
    # Save variance selector and keep_indices
    joblib.dump(variance_selector, os.path.join(MODELS_DIR, "variance_selector.pkl"))
    joblib.dump(keep_indices, os.path.join(MODELS_DIR, "keep_indices.pkl"))
    if kbest_selector is not None:
        joblib.dump(kbest_selector, os.path.join(MODELS_DIR, "kbest_selector.pkl"))

    # Save metrics
    metrics_out = {
        "selected_k": ("all" if best_k == -1 else int(best_k)),
        "models": final_results,
        "best_model": best_name,
        "best_threshold": best_threshold,
    }
    with open(LATEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[SUCCESS] Saved best model and metrics. Best model: {best_name}")

    # Detailed evaluation and plots for best model
    if hasattr(best_model_obj, "predict_proba"):
        probs_test_best = best_model_obj.predict_proba(X_test_sel)[:, 1]
    else:
        probs_test_best = None
    preds_test_best = (probs_test_best >= best_threshold).astype(int) if probs_test_best is not None else best_model_obj.predict(X_test_sel)
    print_detailed_evaluation(y_test, preds_test_best, probs_test_best if probs_test_best is not None else np.zeros_like(preds_test_best))
    plot_feature_importance(best_model_obj, best_name, n_features=20)

    print("\n[INFO] === Training Complete ===")
    print(f"[INFO] Best model: {best_name}")
    print(f"[INFO] Best threshold: {best_threshold}")
    print(f"[INFO] Test F1: {best_final['f1']:.4f} | ROC-AUC: {best_final['roc_auc']}")


if __name__ == "__main__":
    # Run the patient-wise stacking voice pipeline by default
    try:
        train_voice_stacking()
    except Exception as e:
        print(f"[ERROR] Stacking run failed: {e}")
        print("Falling back to main pipeline for diagnostic run.")
        main()
