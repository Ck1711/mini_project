"""Quick voice experiments: regularized models + small ensemble.

Purpose: run faster trials to reduce overfitting and try ensemble without long
RandomizedSearchCV. Uses patient-wise CV (StratifiedGroupKFold) and aggregates
OOF probabilities to tune decision threshold and evaluate final test metrics.
"""
import json
import os
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from patient_data import prepare_voice_split_pipeline, print_voice_diagnostics


RANDOM_STATE = 42


def _sgkf_splits(n_splits: int = 5):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def collect_oof_probs(X, y, groups, model, sample_weight=None):
    sgkf = _sgkf_splits()
    oof = np.zeros(len(y), dtype=np.float64)
    for tr_idx, val_idx in sgkf.split(X, y, groups):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr = y[tr_idx]
        sw = sample_weight[tr_idx] if sample_weight is not None else None
        model.fit(X_tr, y_tr, sample_weight=sw)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
    return oof


def threshold_from_oof(y_true, y_prob):
    best_t, best_bal = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 161):
        bal = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    return best_t, best_bal


def main():
    os.makedirs("models", exist_ok=True)
    print("[INFO] Preparing voice split pipeline (k=100)...")
    vs = prepare_voice_split_pipeline(k_features=100, prune_weak=True)

    train_df = vs.train_df
    test_df = vs.test_df
    patient_col = vs.patient_col
    feature_cols = vs.feature_cols

    X_train = train_df[feature_cols].values
    y_train = train_df["_label"].values.astype(int)
    groups = train_df[patient_col].astype(str).values

    X_test = test_df[feature_cols].values
    y_test = test_df["_label"].values.astype(int)

    print_voice_diagnostics(train_df, test_df, patient_col)

    # scale features (selector already applied in pipeline, but ensure scaling)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join("models", "voice_quick_scaler.pkl"))

    # sample weights
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    skw = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = {int(c): float(w) for c, w in zip(classes, skw)}
    sample_weight = np.array([weight_map[int(lbl)] for lbl in y_train], dtype=np.float32)

    models = {
        "xgb_reg": xgb.XGBClassifier(
            objective="binary:logistic",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            use_label_encoder=False,
            verbosity=0,
        ),
        "histgb": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE
        ),
        "logreg": LogisticRegression(C=0.5, penalty="l2", solver="saga", max_iter=2000, random_state=RANDOM_STATE),
    }

    oof_probs = {}
    for name, mdl in models.items():
        print(f"[INFO] OOF CV for {name}...")
        oof = collect_oof_probs(X_train, y_train, groups, mdl, sample_weight=sample_weight)
        t, bal = threshold_from_oof(y_train, oof)
        print(f"[INFO] {name} OOF balanced acc={bal*100:.2f}% threshold={t:.3f}")
        oof_probs[name] = (oof, t)

    # simple ensemble: average OOF probs
    ensemble_oof = np.mean([v[0] for v in oof_probs.values()], axis=0)
    t_ens, bal_ens = threshold_from_oof(y_train, ensemble_oof)
    print(f"[INFO] Ensemble OOF balanced acc={bal_ens*100:.2f}% threshold={t_ens:.3f}")

    # retrain best model (choose ensemble approach: average preds from retrained members)
    print("[INFO] Retraining individual models on full train and evaluating on test...")
    test_probs = np.zeros(len(X_test), dtype=np.float64)
    for name, mdl in models.items():
        print(f"[INFO] Training {name} on full train...")
        if hasattr(mdl, "fit"):
            mdl.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            mdl.fit(X_train, y_train)
        joblib.dump(mdl, os.path.join("models", f"voice_{name}.pkl"))
        test_probs += mdl.predict_proba(X_test)[:, 1]
    test_probs /= len(models)

    # use ensemble threshold
    thresh = t_ens
    y_pred = (test_probs >= thresh).astype(int)

    acc = accuracy_score(y_test, y_pred)
    bal = balanced_accuracy_score(y_test, y_pred)
    print("\n[INFO] --- Quick Experiment Test Results ---")
    print(f"Accuracy: {acc*100:.2f}% | Balanced accuracy: {bal*100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"F1: {f1_score(y_test, y_pred, zero_division=0)*100:.2f}%")
    print(f"ROC-AUC: {roc_auc_score(y_test, test_probs)*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["healthy", "parkinson"]))
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")


if __name__ == "__main__":
    main()
