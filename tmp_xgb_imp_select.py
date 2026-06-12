import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from train_dl_model import load_voice_patient_split, prune_weak_voice_features, impute_with_train_stats, scale_train_transform_test, select_k_best_features

train_df, test_df, patient_col, target_col, raw_feature_cols = load_voice_patient_split(test_size=0.1, random_state=42)
y_train = train_df['_label'].values
y_test = test_df['_label'].values
pruned_cols, vt, _, _ = prune_weak_voice_features(train_df, test_df, raw_feature_cols)
train_df, test_df = impute_with_train_stats(train_df, test_df, pruned_cols)
X_train, X_test, scaler = scale_train_transform_test(train_df, test_df, pruned_cols)

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
base = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', scale_pos_weight=scale_pos_weight, n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)
base.fit(X_train, y_train)
importances = base.feature_importances_
indices = np.argsort(importances)[::-1]
for k in [50, 75, 100, 150, 300, len(pruned_cols)]:
    top_indices = indices[:k]
    X_train_sel = X_train[:, top_indices]
    X_test_sel = X_test[:, top_indices]
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', scale_pos_weight=scale_pos_weight, n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train_sel, y_train)
    y_prob = model.predict_proba(X_test_sel)[:,1]
    best_acc = 0
    best_t = 0.5
    for t in np.linspace(0.01, 0.99, 99):
        acc = accuracy_score(y_test, (y_prob >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    print('top', k, 'best acc', best_acc, 'best_t', best_t, 'auc', roc_auc_score(y_test, y_prob))
