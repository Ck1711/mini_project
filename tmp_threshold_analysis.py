import numpy as np
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from train_dl_model import select_best_k_group_cv
from patient_data import (
    load_voice_frame_split,
    prune_weak_voice_features,
    impute_with_train_stats,
    scale_train_transform_test,
    select_k_best_features,
    print_class_distribution_and_weights,
)

train_df, test_df, patient_col, target_col, raw_feature_cols = load_voice_frame_split(test_size=0.1, random_state=42)
y_train = train_df["_label"].values
groups_train = train_df[patient_col].astype(str).values
weight_map, scale_pos_weight = print_class_distribution_and_weights(y_train, "train", print_weights=True)
pruned_cols, vt, _, _ = prune_weak_voice_features(train_df, test_df, raw_feature_cols)
train_df, test_df = impute_with_train_stats(train_df, test_df, pruned_cols)
X_train, X_test, scaler = scale_train_transform_test(train_df, test_df, pruned_cols)

k = 200
X_train_sel, X_test_sel, selector, feature_cols = select_k_best_features(X_train, y_train, X_test, pruned_cols, k)
print('Selected features:', len(feature_cols))

params = {
    'objective':'binary:logistic',
    'eval_metric':'auc',
    'scale_pos_weight':scale_pos_weight,
    'n_estimators':1000,
    'max_depth':4,
    'learning_rate':0.03,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'colsample_bylevel':0.8,
    'min_child_weight':3,
    'reg_alpha':0.3,
    'reg_lambda':1.5,
    'gamma':0.1,
    'grow_policy':'lossguide',
    'max_leaves':32,
    'tree_method':'hist',
    'random_state':42,
    'verbosity':0,
    'n_jobs':-1,
}
clf = xgb.XGBClassifier(**params)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y_train), dtype=float)
for tr_idx, val_idx in sgkf.split(X_train_sel, y_train, groups_train):
    m = clone(clf)
    m.set_params(n_estimators=200)
    m.fit(X_train_sel[tr_idx], y_train[tr_idx], verbose=False)
    oof[val_idx] = m.predict_proba(X_train_sel[val_idx])[:, 1]

best_f1 = (0, 0)
best_bal = (0, 0)
best_acc = (0, 0)
for t in np.linspace(0.05, 0.95, 181):
    pred = (oof >= t).astype(int)
    f1 = f1_score(y_train, pred, zero_division=0)
    bal = balanced_accuracy_score(y_train, pred)
    acc = accuracy_score(y_train, pred)
    if f1 > best_f1[0]:
        best_f1 = (f1, t)
    if bal > best_bal[0]:
        best_bal = (bal, t)
    if acc > best_acc[0]:
        best_acc = (acc, t)
print('best_f1', best_f1)
print('best_bal', best_bal)
print('best_acc', best_acc)

clf.set_params(n_estimators=1000)
clf.fit(X_train_sel, y_train)
y_prob = clf.predict_proba(X_test_sel)[:, 1]
for name, t in [('f1', best_f1[1]), ('bal', best_bal[1]), ('acc', best_acc[1]), ('0.5', 0.5)]:
    pred = (y_prob >= t).astype(int)
    print(name, t, 'acc', accuracy_score(y_test, pred), 'bal', balanced_accuracy_score(y_test, pred), 'f1', f1_score(y_test, pred, zero_division=0))
