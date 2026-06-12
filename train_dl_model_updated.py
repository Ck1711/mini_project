"""
Voice branch: XGBoost Stacking with OOF Meta-learner + Optuna HP Search
Leak-free patient-wise splits, aggressive feature engineering, and robust threshold tuning.

IMPROVED STRATEGY:
  - Group-aware CV prevents patient leakage
  - Aggressive feature pruning: variance + correlation filtering
  - Diverse base ensemble: 3×XGBoost + ExtraTrees + CatBoost
  - OOF stacking reduces variance on small test sets
  - Meta-learner (LogisticRegression) combines OOF predictions
  - Optuna-based hyperparameter optimization for ensemble
  - Threshold tuned to maximize balanced accuracy
  - Leakage detection and reporting
"""
import json
import os
from typing import Dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import random
import time

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import clone

from patient_data import (
    SELECTED_FEATURES_PATH,
    impute_with_train_stats,
    load_voice_frame_split,
    make_selected_voice_frames,
    print_class_distribution_and_weights,
    prune_weak_voice_features,
    save_voice_splits,
    scale_train_transform_test,
    select_k_best_features,
    warn_suspicious_accuracy,
)

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    HAS_CATBOOST = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    optuna = None
    HAS_OPTUNA = False

# ── Config ──────────────────────────────────────────────────────────────────
K_CANDIDATES = [100, 150, 200, 300, 400, 500, 700, 1000]
CV_FOLDS = 5
RANDOM_STATE = 42
OVERFIT_GAP_THRESHOLD = 0.08
CORRELATION_THRESHOLD = 0.90
OPTUNA_TRIALS = 100
THRESHOLD_OPTIMIZATION_WEIGHT = 0.6  # F1 weight when combining with balanced accuracy

# ── Paths ────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

xgb_base_models_path  = os.path.join("models", "voice_xgb_base_models.pkl")
scaler_save_path       = os.path.join("models", "scaler.pkl")
selector_save_path     = os.path.join("models", "feature_selector.pkl")
variance_selector_path = os.path.join("models", "voice_variance_selector.pkl")
pruned_cols_path       = os.path.join("models", "voice_pruned_columns.pkl")
impute_means_path      = os.path.join("models", "voice_impute_means.pkl")
best_k_path            = os.path.join("models", "voice_best_k.json")
threshold_path         = os.path.join("models", "voice_decision_threshold.json")
selected_features_path = SELECTED_FEATURES_PATH


# ── Helpers ──────────────────────────────────────────────────────────────────
def _sgkf(n_splits=CV_FOLDS):
    """Return StratifiedGroupKFold for patient-wise group-aware splitting."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def aggregate_group_predictions(predictions: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate row-level probability predictions into group-level averages."""
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    df = pd.DataFrame(predictions, columns=[f"m{i}" for i in range(predictions.shape[1])])
    df["group"] = groups.astype(str)
    grouped = df.groupby("group").mean()
    return grouped.values, grouped.index.to_numpy()


def group_labels(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one label per group using the first row encountered for each group."""
    df = pd.DataFrame({"group": groups.astype(str), "label": y.astype(int)})
    grouped = df.groupby("group").first()
    return grouped["label"].values, grouped.index.to_numpy()


def sample_weights_from_map(y: np.ndarray, weight_map: Dict[int, float]) -> np.ndarray:
    """Convert label-to-weight dict into sample weight array."""
    return np.array([weight_map[int(l)] for l in y], dtype=np.float32)


# ── Additional helpers ──────────────────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def patient_group_metrics(y_true: np.ndarray, y_prob: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    group_prob, _ = aggregate_group_predictions(y_prob, groups)
    y_group, _ = group_labels(y_true, groups)
    y_pred = (group_prob.ravel() >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_group, group_prob.ravel())),
        "accuracy": float(accuracy_score(y_group, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_group, y_pred)),
        "f1": float(f1_score(y_group, y_pred, zero_division=0)),
    }


def compare_feature_selectors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    k_candidates,
    scale_pos_weight: float,
) -> tuple[str, int, Dict[str, Dict[int, float]]]:
    methods = {
        "f_classif": f_classif,
        "mutual_info_classif": mutual_info_classif,
    }
    best_method = None
    best_k = None
    best_score = -1.0
    scores = {m: {} for m in methods}
    print("\n[INFO] === Comparing SelectKBest feature selection methods ===")

    for method_name, score_func in methods.items():
        for k in k_candidates:
            k_eff = min(k, X_train.shape[1])
            fold_scores = []
            sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            for tr_idx, val_idx in sgkf.split(X_train, y_train, groups):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                groups_val = groups[val_idx]

                selector = SelectKBest(score_func=score_func, k=k_eff)
                X_tr_sel = selector.fit_transform(X_tr, y_tr)
                X_val_sel = selector.transform(X_val)

                clf = _build_xgb(scale_pos_weight)
                clf.set_params(n_estimators=200)
                clf.fit(X_tr_sel, y_tr, verbose=False)
                y_val_prob = clf.predict_proba(X_val_sel)[:, 1]
                group_prob, _ = aggregate_group_predictions(y_val_prob, groups_val)
                y_group, _ = group_labels(y_val, groups_val)
                fold_scores.append(roc_auc_score(y_group, group_prob.ravel()))

            mean_score = float(np.mean(fold_scores))
            scores[method_name][k] = mean_score
            print(
                f"[INFO]   method={method_name} k={k:4d} | patient ROC-AUC={mean_score*100:.2f}%"
            )
            if mean_score > best_score:
                best_score = mean_score
                best_method = method_name
                best_k = k

    print(
        f"[SUCCESS] Selected {best_method} with k={best_k} (patient ROC-AUC={best_score*100:.2f}%)"
    )
    return best_method, best_k, scores


# ── XGBoost model builder ────────────────────────────────────────────────────
def _build_xgb(scale_pos_weight: float) -> xgb.XGBClassifier:
    """Build a strong baseline XGBoost classifier."""
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        min_child_weight=3,
        reg_alpha=0.3,
        reg_lambda=1.5,
        gamma=0.1,
        grow_policy="lossguide",
        max_leaves=32,
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )


def _build_xgb_variants(scale_pos_weight: float) -> list:
    """Create an XGBoost-only ensemble of diverse hyperparameter variants."""
    return [
        _build_xgb(scale_pos_weight),
        xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            n_estimators=1200,
            max_depth=5,
            learning_rate=0.025,
            subsample=0.9,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_alpha=0.5,
            reg_lambda=1.0,
            gamma=0.05,
            grow_policy="lossguide",
            tree_method="hist",
            random_state=RANDOM_STATE + 1,
            verbosity=0,
            n_jobs=-1,
        ),
        xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            n_estimators=1400,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.75,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.2,
            reg_lambda=2.0,
            gamma=0.2,
            grow_policy="lossguide",
            tree_method="hist",
            random_state=RANDOM_STATE + 2,
            verbosity=0,
            n_jobs=-1,
        ),
    ]


# ── SelectKBest search with XGBoost ─────────────────────────────────────────
def select_best_k_group_cv(
    X_train, y_train, groups, pruned_cols, scale_pos_weight, k_candidates
):
    """
    Select best k features using group-aware CV with XGBoost.
    Prevents patient leakage via StratifiedGroupKFold.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_bal = {k: [] for k in k_candidates}
    fold_auc = {k: [] for k in k_candidates}
    print("\n[INFO] === SelectKBest k search (XGBoost, group-aware CV) ===")

    for tr_idx, val_idx in sgkf.split(X_train, y_train, groups):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        groups_val = groups[val_idx]

        for k in k_candidates:
            k_eff = min(k, X_tr.shape[1])
            sel = SelectKBest(score_func=f_classif, k=k_eff)
            X_tr_s = sel.fit_transform(X_tr, y_tr)
            X_val_s = sel.transform(X_val)

            clf = _build_xgb(scale_pos_weight)
            clf.set_params(n_estimators=200)
            try:
                clf.fit(X_tr_s, y_tr, verbose=False)
                y_val_proba = clf.predict_proba(X_val_s)[:, 1]
                y_val_group, _ = group_labels(y_val, groups_val)
                y_val_group_pred, _ = aggregate_group_predictions(y_val_proba, groups_val)
                auc = roc_auc_score(y_val_group, y_val_group_pred.ravel())
                bal = balanced_accuracy_score(y_val_group, (y_val_group_pred.ravel() >= 0.5).astype(int))
            except Exception:
                y_val_pred = clf.predict(X_val_s)
                y_val_group, _ = group_labels(y_val, groups_val)
                y_val_group_pred, _ = aggregate_group_predictions(y_val_pred, groups_val)
                auc = roc_auc_score(y_val_group, y_val_group_pred.ravel())
                bal = balanced_accuracy_score(y_val_group, (y_val_group_pred.ravel() >= 0.5).astype(int))

            fold_auc[k].append(auc)
            fold_bal[k].append(bal)

    mean_bal = {k: float(np.mean(fold_bal[k])) for k in k_candidates}
    mean_auc = {k: float(np.mean(fold_auc[k])) for k in k_candidates}
    for k in k_candidates:
        print(
            f"[INFO]   k={k:3d} | mean bal={mean_bal[k]*100:.2f}% ± {np.std(fold_bal[k])*100:.2f}% |"
            f" mean AUC={mean_auc[k]*100:.2f}% ± {np.std(fold_auc[k])*100:.2f}%"
        )

    best_k = max(mean_bal, key=mean_bal.get)
    print(
        f"[SUCCESS] Best k={best_k} with mean balanced accuracy={mean_bal[best_k]*100:.2f}%"
    )
    return best_k, mean_bal


# ── OOF collection with ensemble ────────────────────────────────────────────
def collect_oof_stack(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    base_models: list,
    sample_weight: np.ndarray,
) -> np.ndarray:
    """
    Collect out-of-fold probabilities from multiple base models.
    Returns shape (n_train, n_models) with OOF probabilities.
    """
    sgkf = _sgkf()
    oof = np.zeros((len(y_train), len(base_models)), dtype=np.float64)

    print("\n[INFO] === Collecting OOF predictions (ensemble, group-aware) ===")
    for fold_i, (tr_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups), 1):
        print(f"[INFO]   Fold {fold_i}/{CV_FOLDS}...", end=" ")
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]
        sw_tr = sample_weight[tr_idx] if sample_weight is not None else None

        for m_i, model in enumerate(base_models):
            m = clone(model)
            # Use early stopping on the fold's validation set to improve generalization
            fit_kwargs = {}
            if isinstance(m, xgb.XGBClassifier):
                fit_kwargs.update({"eval_set": [(X_val, y_val)], "early_stopping_rounds": 50, "verbose": False})
            # CatBoost may accept verbose=0 instead of our xgb kwargs
            if HAS_CATBOOST and isinstance(m, CatBoostClassifier):
                fit_kwargs = {k: v for k, v in fit_kwargs.items() if k != "eval_set"}
                fit_kwargs.update({"verbose": 0})
            try:
                m.fit(X_tr, y_tr, sample_weight=sw_tr, **fit_kwargs)
            except TypeError:
                # Some estimators don't accept sample_weight or our kwargs
                try:
                    m.fit(X_tr, y_tr, **fit_kwargs)
                except TypeError:
                    m.fit(X_tr, y_tr)
            # handle models without predict_proba
            if hasattr(m, "predict_proba"):
                oof[val_idx, m_i] = m.predict_proba(X_val)[:, 1]
            else:
                oof[val_idx, m_i] = m.predict(X_val)
            print("✓", end=" ")
        print()

    for m_i, _ in enumerate(base_models):
        try:
            auc = roc_auc_score(y_train, oof[:, m_i])
            print(f"[INFO]   OOF AUC model {m_i+1}: {auc*100:.2f}%")
        except Exception:
            pass

    return oof


def train_base_models_full(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_models: list,
    sample_weight: np.ndarray,
    groups: np.ndarray = None,
):
    """Retrain each base model on the full training set with internal validation split for early stopping."""
    from sklearn.model_selection import GroupShuffleSplit

    trained = []
    print("\n[INFO] === Retraining ensemble on full train set ===")
    # create small patient-level holdout for early stopping
    X_tr_full, X_val_full, y_tr_full, y_val_full, sw_tr_full, sw_val_full = None, None, None, None, None, None
    try:
        if groups is not None and len(groups) == len(y_train):
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
            tr_idx, val_idx = next(splitter.split(X_train, y_train, groups))
            X_tr_full, X_val_full = X_train[tr_idx], X_train[val_idx]
            y_tr_full, y_val_full = y_train[tr_idx], y_train[val_idx]
            if sample_weight is not None and len(sample_weight) == len(y_train):
                sw_tr_full, sw_val_full = sample_weight[tr_idx], sample_weight[val_idx]
        else:
            raise ValueError("Group labels unavailable for patient-aware holdout")
    except Exception:
        X_tr_full, y_tr_full = X_train, y_train

    for model in base_models:
        m = clone(model)
        fit_kwargs = {}
        if isinstance(m, xgb.XGBClassifier) and X_val_full is not None:
            fit_kwargs.update(
                {"eval_set": [(X_val_full, y_val_full)], "early_stopping_rounds": 50, "verbose": False}
            )
        try:
            if X_tr_full is not None and X_tr_full is not X_train:
                m.fit(
                    X_tr_full,
                    y_tr_full,
                    sample_weight=(sw_tr_full if sw_tr_full is not None else None),
                    **fit_kwargs,
                )
            else:
                m.fit(X_train, y_train, sample_weight=sample_weight, **fit_kwargs)
        except TypeError:
            try:
                if X_tr_full is not None and X_tr_full is not X_train:
                    m.fit(X_tr_full, y_tr_full, **fit_kwargs)
                else:
                    m.fit(X_train, y_train, **fit_kwargs)
            except TypeError:
                m.fit(X_train, y_train)
        trained.append(m)
        print(f"[INFO]   {type(m).__name__} trained ✓")
    return trained


def predict_test_stack(
    trained_base,
    X_test: np.ndarray,
) -> np.ndarray:
    """Get test probabilities from all trained models → shape (n_test, n_models)."""
    cols = []
    for m in trained_base:
        if hasattr(m, "predict_proba"):
            cols.append(m.predict_proba(X_test)[:, 1])
        else:
            cols.append(m.predict(X_test))
    return np.column_stack(cols)


def tune_decision_threshold(y_true, y_prob):
    """
    Find optimal decision threshold by maximizing patient-level balanced accuracy.
    This favors a more generalizable threshold on imbalanced voice data.
    """
    best_t, best_bal, best_acc = 0.5, 0.0, 0.0
    for t in np.linspace(0.05, 0.95, 361):
        pred = (y_prob >= t).astype(int)
        bal = balanced_accuracy_score(y_true, pred)
        acc = accuracy_score(y_true, pred)
        if bal > best_bal or (bal == best_bal and acc > best_acc):
            best_bal, best_acc, best_t = bal, acc, float(t)
    return best_t, best_bal


def check_overfitting(train_acc, val_acc):
    gap = train_acc - val_acc
    print(f"\n[INFO] Train acc: {train_acc*100:.2f}% | Test acc: {val_acc*100:.2f}% | Gap: {gap*100:.2f}%")
    if gap > OVERFIT_GAP_THRESHOLD:
        print("[WARNING] Overfitting detected (gap > 8%)")


# ── Optuna hyperparameter search ──────────────────────────────────────────
def hp_search_xgb(
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    scale_pos_weight: float,
    n_trials: int = 50,
):
    """
    Optuna-based hyperparameter search for XGBoost, CatBoost, ExtraTrees ensemble.
    Optimizes patient-level AUC using group-aware CV.
    """
    if not HAS_OPTUNA:
        print("[WARNING] Optuna not installed; skipping hyperparameter search")
        return None

    print(f"\n[INFO] === Optuna HP Search ({n_trials} trials, optimizing patient-level AUC) ===")
    sgkf = _sgkf()
    sample_weight = sample_weights_from_map(y_train, {0: 1.92, 1: 0.676})

    def objective(trial):
        """Optuna objective: maximize patient-level ROC-AUC."""
        model_choice = trial.suggest_categorical("model_type", ["xgb", "catboost", "extratrees"])

        if model_choice == "xgb":
            params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1500, step=100),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
                "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("xgb_alpha", 0.0, 5.0, log=True),
                "reg_lambda": trial.suggest_float("xgb_lambda", 0.0, 5.0, log=True),
                "gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
            }
            model = xgb.XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=scale_pos_weight,
                grow_policy="lossguide",
                tree_method="hist",
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
            )

        elif model_choice == "catboost" and HAS_CATBOOST:
            params = {
                "iterations": trial.suggest_int("cb_iterations", 300, 1000, step=100),
                "depth": trial.suggest_int("cb_depth", 3, 10),
                "learning_rate": trial.suggest_float("cb_lr", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("cb_l2", 1e-3, 10.0, log=True),
                "border_count": trial.suggest_int("cb_border_count", 32, 256, step=32),
            }
            model = CatBoostClassifier(
                **params,
                random_state=RANDOM_STATE,
                verbose=0,
            )

        else:  # ExtraTrees
            params = {
                "n_estimators": trial.suggest_int("et_n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_int("et_max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("et_min_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("et_min_leaf", 1, 10),
                "max_features": trial.suggest_categorical("et_max_features", ["sqrt", "log2"]),
            }
            model = ExtraTreesClassifier(
                **params,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

        # Evaluate using group-aware CV
        fold_aucs = []
        for tr_idx, val_idx in sgkf.split(X_train_sel, y_train, groups):
            X_tr, X_val = X_train_sel[tr_idx], X_train_sel[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            groups_val = groups[val_idx]
            sw_tr = sample_weight[tr_idx]

            m = clone(model)
            fit_kwargs = {}
            if isinstance(m, xgb.XGBClassifier):
                fit_kwargs = {"verbose": False}
            elif HAS_CATBOOST and isinstance(m, CatBoostClassifier):
                fit_kwargs = {"verbose": 0}

            try:
                m.fit(X_tr, y_tr, sample_weight=sw_tr, **fit_kwargs)
            except TypeError:
                m.fit(X_tr, y_tr, **fit_kwargs)

            # Patient-level evaluation
            if hasattr(m, "predict_proba"):
                y_val_prob = m.predict_proba(X_val)[:, 1]
            else:
                y_val_prob = m.predict(X_val)

            y_val_group, _ = group_labels(y_val, groups_val)
            y_val_group_prob, _ = aggregate_group_predictions(y_val_prob, groups_val)
            auc = roc_auc_score(y_val_group, y_val_group_prob.ravel())
            fold_aucs.append(auc)

        mean_auc = float(np.mean(fold_aucs))
        return mean_auc

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    print(f"\n[SUCCESS] Best trial #{best_trial.number}")
    print(f"[INFO] Best patient-level AUC: {best_trial.value*100:.2f}%")
    print(f"[INFO] Best params: {best_trial.params}")

    # Save results to JSON
    hp_results_path = os.path.join("outputs", "optuna_hp_search_results.json")
    with open(hp_results_path, "w") as f:
        json.dump({
            "best_trial": best_trial.number,
            "best_auc": best_trial.value,
            "best_params": best_trial.params,
            "n_trials": n_trials,
        }, f, indent=2)
    print(f"[SUCCESS] HP search results saved to {hp_results_path}")

    return {
        "best_trial": best_trial,
        "best_params": best_trial.params,
        "best_auc": best_trial.value,
        "study": study,
    }


class VoicePipeline:
    def __init__(
        self,
        force_k: int = None,
        hp_search: bool = False,
        hp_trials: int = 12,
        test_size: float = 0.1,
    ):
        self.force_k = force_k
        self.hp_search = hp_search
        self.hp_trials = hp_trials
        self.test_size = test_size
        self._reset()

    def _reset(self):
        self.train_df = None
        self.test_df = None
        self.patient_col = None
        self.raw_feature_cols = None
        self.pruned_cols = None
        self.vt = None
        self.selector = None
        self.feature_cols = None
        self.X_train = None
        self.X_test = None
        self.X_train_sel = None
        self.X_test_sel = None
        self.y_train = None
        self.y_test = None
        self.groups_train = None
        self.groups_test = None
        self.weight_map = None
        self.scale_pos_weight = None
        self.sample_weight = None
        self.base_models = None
        self.oof_train = None
        self.meta = None
        self.decision_threshold = None

    def run(self):
        self.load_data()
        self.prepare_features()
        self.select_features()
        if self.hp_search:
            return self.run_hp_search()
        self.build_models()
        self.collect_oof()
        self.train_meta_learner()
        self.retrain_ensemble()
        self.evaluate()
        self.save_artifacts()
        self.save_plots()
        return self

    def load_data(self):
        self.train_df, self.test_df, self.patient_col, _target_col, self.raw_feature_cols = (
            load_voice_frame_split(test_size=self.test_size, random_state=RANDOM_STATE)
        )
        save_voice_splits(self.train_df, self.test_df, self.patient_col)
        self.y_train = self.train_df["_label"].values
        self.y_test = self.test_df["_label"].values
        self.groups_train = self.train_df[self.patient_col].astype(str).values
        self.groups_test = self.test_df[self.patient_col].astype(str).values
        self._log_split_stats()
        self._confirm_no_leakage()
        self.weight_map, self.scale_pos_weight = print_class_distribution_and_weights(
            self.y_train, "train", print_weights=True
        )
        print_class_distribution_and_weights(self.y_test, "test", print_weights=False)

    def prepare_features(self):
        print("\n[INFO] === Feature Engineering ===")
        self.pruned_cols, self.vt, _, _ = prune_weak_voice_features(
            self.train_df, self.test_df, self.raw_feature_cols
        )
        self.train_df, self.test_df = impute_with_train_stats(
            self.train_df, self.test_df, self.pruned_cols
        )
        joblib.dump(self.train_df[self.pruned_cols].mean(), impute_means_path)
        self.X_train, self.X_test, self.scaler = scale_train_transform_test(
            self.train_df, self.test_df, self.pruned_cols
        )

    def select_features(self):
        print("\n[INFO] === Feature Selection ===")
        if self.force_k is None:
            self.best_k, self.k_scores = select_best_k_group_cv(
                self.X_train,
                self.y_train,
                self.groups_train,
                self.pruned_cols,
                self.scale_pos_weight,
                K_CANDIDATES,
            )
        else:
            self.best_k = int(self.force_k)
            self.k_scores = {k: None for k in K_CANDIDATES}
            print(f"[INFO] Forcing feature selection k={self.best_k} (skipping CV)")

        self.X_train_sel, self.X_test_sel, self.selector, self.feature_cols = select_k_best_features(
            self.X_train, self.y_train, self.X_test, self.pruned_cols, self.best_k
        )

        self.train_sel, self.test_sel = make_selected_voice_frames(
            self.X_train_sel,
            self.X_test_sel,
            self.feature_cols,
            self.train_df,
            self.test_df,
            self.patient_col,
        )

        with open(best_k_path, "w") as f:
            json.dump({"best_k": self.best_k, "cv_mean_auc": self.k_scores}, f, indent=2)
        with open(selected_features_path, "w") as f:
            f.writelines(f"{n}\n" for n in self.feature_cols)
        print(f"[SUCCESS] Saved best k={self.best_k}, {len(self.feature_cols)} features.")

    def build_models(self):
        print("\n[INFO] === Ensemble OOF Stacking ===")
        self.sample_weight = sample_weights_from_map(self.y_train, self.weight_map)
        # start with diverse XGBoost variants
        xgb_models = self._build_xgb_variants(self.scale_pos_weight)
        self.base_models = list(xgb_models)

        # add an ExtraTrees variant for diversity
        et = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.base_models.append(et)

        # add CatBoost if available (optional, environment-dependent)
        if HAS_CATBOOST:
            try:
                cb = CatBoostClassifier(
                    iterations=600,
                    depth=6,
                    learning_rate=0.03,
                    random_state=RANDOM_STATE,
                    verbose=0,
                )
                self.base_models.append(cb)
            except Exception:
                pass

        print(f"[INFO] Using {len(self.base_models)} base models for OOF stacking: "
              f"{', '.join(type(m).__name__ for m in self.base_models)}")

    def collect_oof(self):
        self.oof_train = collect_oof_stack(
            self.X_train_sel,
            self.y_train,
            self.groups_train,
            self.base_models,
            self.sample_weight,
        )
        self.oof_train_patient, _ = aggregate_group_predictions(
            self.oof_train, self.groups_train
        )
        self.y_train_patient, _ = group_labels(self.y_train, self.groups_train)

    def train_meta_learner(self):
        print("\n[INFO] === Training Meta-Learner (LogisticRegression) ===")
        self.meta = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
        self.meta.fit(self.oof_train_patient, self.y_train_patient)
        self.oof_meta_prob = self.meta.predict_proba(self.oof_train_patient)[:, 1]
        self.oof_meta_auc = roc_auc_score(self.y_train_patient, self.oof_meta_prob)
        print(f"[SUCCESS] Meta-learner OOF AUC: {self.oof_meta_auc*100:.2f}%")
        self.decision_threshold, self.oof_bal = tune_decision_threshold(
            self.y_train_patient, self.oof_meta_prob
        )
        print(
            f"[SUCCESS] Decision threshold={self.decision_threshold:.3f} | "
            f"OOF bal-acc={self.oof_bal*100:.2f}%"
        )
        with open(threshold_path, "w") as f:
            json.dump(
                {"threshold": self.decision_threshold, "oof_balanced_accuracy": self.oof_bal},
                f,
                indent=2,
            )

    def retrain_ensemble(self):
        self.trained_base = train_base_models_full(
            self.X_train_sel,
            self.y_train,
            self.base_models,
            self.sample_weight,
            self.groups_train,
        )

    def evaluate(self):
        train_stack = predict_test_stack(self.trained_base, self.X_train_sel)
        test_stack = predict_test_stack(self.trained_base, self.X_test_sel)
        train_stack_patient, _ = aggregate_group_predictions(train_stack, self.groups_train)
        test_stack_patient, _ = aggregate_group_predictions(test_stack, self.groups_test)

        self.y_prob_train = self.meta.predict_proba(train_stack_patient)[:, 1]
        self.y_prob = self.meta.predict_proba(test_stack_patient)[:, 1]
        self.y_pred_train = (self.y_prob_train >= self.decision_threshold).astype(int)
        self.y_pred = (self.y_prob >= self.decision_threshold).astype(int)

        self.y_train_patient_label, _ = group_labels(self.y_train, self.groups_train)
        self.y_test_patient_label, _ = group_labels(self.y_test, self.groups_test)

        self.train_acc = accuracy_score(self.y_train_patient_label, self.y_pred_train)
        self.test_acc = accuracy_score(self.y_test_patient_label, self.y_pred)
        self.bal_acc = balanced_accuracy_score(self.y_test_patient_label, self.y_pred)
        self.test_auc = roc_auc_score(self.y_test_patient_label, self.y_prob)

        check_overfitting(self.train_acc, self.test_acc)
        self._report_results()

    def run_hp_search(self):
        print("[INFO] Running HP search mode: loading data and selecting features...")
        print("[INFO] Optuna will search hyperparameters for XGBoost, CatBoost, and ExtraTrees...")
        return hp_search_xgb(
            self.X_train_sel,
            self.y_train,
            self.groups_train,
            self.scale_pos_weight,
            n_trials=self.hp_trials,
        )

    def save_artifacts(self):
        joblib.dump(self.trained_base, xgb_base_models_path)
        joblib.dump(self.scaler, scaler_save_path)
        joblib.dump(self.selector, selector_save_path)
        joblib.dump(self.vt, variance_selector_path)
        joblib.dump(self.pruned_cols, pruned_cols_path)
        joblib.dump(self.meta, os.path.join("models", "voice_meta_learner.pkl"))
        joblib.dump(self.train_sel, os.path.join("models", "voice_train_features.pkl"))
        joblib.dump(self.test_sel, os.path.join("models", "voice_test_features.pkl"))
        print("[SUCCESS] All models saved.")

    def save_plots(self):
        cm = confusion_matrix(self.y_test_patient_label, self.y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["healthy", "parkinson"],
            yticklabels=["healthy", "parkinson"],
        )
        plt.title("Voice Confusion Matrix (Ensemble + OOF Meta-Learner)")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.savefig(os.path.join("outputs", "voice_confusion_matrix.png"))
        plt.close()

        fpr, tpr, _ = roc_curve(self.y_test_patient_label, self.y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC={self.test_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.legend()
        plt.title("Voice ROC (Ensemble + OOF Meta-Learner)")
        plt.savefig(os.path.join("outputs", "voice_roc_curve.png"))
        plt.close()

        print_top_xgb_features(self.trained_base, self.feature_cols)
        optional_shap_analysis(self.trained_base[0], self.X_train_sel, self.feature_cols)

    def _confirm_no_leakage(self):
        overlap = set(self.groups_train) & set(self.test_df[self.patient_col].astype(str).values)
        if overlap:
            raise ValueError(f"Patient overlap detected: {sorted(overlap)[:10]}")
        print("[SUCCESS] No patient overlap between voice train and test.")

    def _log_split_stats(self):
        print("[INFO] === Voice split diagnostics ===")
        print(f"[INFO] Train patients: {len(np.unique(self.groups_train))} | Test patients: {len(np.unique(self.groups_test))}")
        print(f"[INFO] Train rows: {len(self.train_df)} | Test rows: {len(self.test_df)}")

    def _report_results(self):
        print("\n[INFO] --- Voice Test Results (patient hold-out, Ensemble + OOF) ---")
        print(f"[SUCCESS] Accuracy:          {self.test_acc*100:.2f}%")
        print(f"[INFO] Balanced accuracy:   {self.bal_acc*100:.2f}%")
        print(f"[INFO] Precision:           {precision_score(self.y_test_patient_label, self.y_pred, zero_division=0)*100:.2f}%")
        print(f"[INFO] Recall:              {recall_score(self.y_test_patient_label, self.y_pred, zero_division=0)*100:.2f}%")
        print(f"[INFO] F1:                  {f1_score(self.y_test_patient_label, self.y_pred, zero_division=0)*100:.2f}%")
        print(f"[INFO] ROC-AUC:             {self.test_auc*100:.2f}%")
        print(f"\n[INFO] Predicted healthy:   {(self.y_pred==0).sum()}")
        print(f"[INFO] Predicted parkinson: {(self.y_pred==1).sum()}")
        print(classification_report(self.y_test_patient_label, self.y_pred, target_names=["healthy", "parkinson"]))
        print(f"[INFO] Confusion matrix:\n{confusion_matrix(self.y_test_patient_label, self.y_pred)}")
        warn_suspicious_accuracy("Voice (Ensemble+OOF)", self.test_acc)

    def _build_xgb(self, scale_pos_weight: float, params: dict = None) -> xgb.XGBClassifier:
        params = params or {}
        default = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": scale_pos_weight,
            "n_estimators": 1000,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "colsample_bylevel": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.3,
            "reg_lambda": 1.5,
            "gamma": 0.1,
            "grow_policy": "lossguide",
            "max_leaves": 32,
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "verbosity": 0,
            "n_jobs": -1,
        }
        default.update(params)
        return xgb.XGBClassifier(**default)

    def _build_xgb_variants(self, scale_pos_weight: float) -> list:
        return [
            self._build_xgb(scale_pos_weight),
            self._build_xgb(
                scale_pos_weight,
                {
                    "n_estimators": 1200,
                    "max_depth": 5,
                    "learning_rate": 0.025,
                    "subsample": 0.9,
                    "colsample_bytree": 0.85,
                    "min_child_weight": 2,
                    "reg_alpha": 0.5,
                    "reg_lambda": 1.0,
                    "gamma": 0.05,
                    "random_state": RANDOM_STATE + 1,
                },
            ),
            self._build_xgb(
                scale_pos_weight,
                {
                    "n_estimators": 1400,
                    "max_depth": 3,
                    "learning_rate": 0.04,
                    "subsample": 0.75,
                    "colsample_bytree": 0.7,
                    "min_child_weight": 5,
                    "reg_alpha": 0.2,
                    "reg_lambda": 2.0,
                    "gamma": 0.2,
                    "random_state": RANDOM_STATE + 2,
                },
            ),
        ]


def print_top_xgb_features(model, feature_cols, top_n=20):
    """Display top feature importances from XGBoost model or first model in list."""
    if isinstance(model, list):
        model = model[0] if model else None
    if model is None or not hasattr(model, "feature_importances_"):
        print(f"[WARNING] Model does not have feature_importances_")
        return

    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    print(f"\n[INFO] Top {top_n} features (from first ensemble member):")
    for rank, i in enumerate(idx, 1):
        print(f"  {rank:2d}. {feature_cols[i]:40s} {imp[i]:.4f}")


def optional_shap_analysis(model, X_train, feature_cols):
    try:
        import shap
        print("[INFO] Running SHAP ...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        plt.figure()
        shap.summary_plot(shap_values, X_train, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join("outputs", "voice_shap_summary.png"))
        plt.close()
        print("[SUCCESS] SHAP saved.")
    except Exception as ex:
        print(f"[WARNING] SHAP skipped: {ex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-k", type=int, default=None, help="Force SelectKBest k (skip CV)")
    parser.add_argument("--hp-search", type=int, default=0, help="Run Optuna HP search with N trials (0=skip)")
    args = parser.parse_args()
    
    # If hp_search is provided as a number, use it; otherwise treat as boolean
    hp_search_enabled = args.hp_search > 0
    hp_trials = args.hp_search if args.hp_search > 0 else 50
    
    pipeline = VoicePipeline(force_k=args.force_k, hp_search=hp_search_enabled, hp_trials=hp_trials)
    pipeline.run()
