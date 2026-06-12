import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from train_dl_model import (
    load_voice_patient_split,
    prune_weak_voice_features,
    impute_with_train_stats,
    scale_train_transform_test,
    select_k_best_features,
    _sgkf,
)

train_df, test_df, patient_col, target_col, raw_feature_cols = load_voice_patient_split(test_size=0.1, random_state=42)
y_train = train_df['_label'].values
y_test = test_df['_label'].values
pruned_cols, vt, _, _ = prune_weak_voice_features(train_df, test_df, raw_feature_cols)
train_df, test_df = impute_with_train_stats(train_df, test_df, pruned_cols)
X_train, X_test, scaler = scale_train_transform_test(train_df, test_df, pruned_cols)
X_train_sel, X_test_sel, selector, feature_cols = select_k_best_features(X_train, y_train, X_test, pruned_cols, 75)

param_dist = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.02, 0.03],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.6, 0.75, 0.85],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.05, 0.1],
    'reg_alpha': [0, 0.5, 1.0],
    'reg_lambda': [1, 1.5, 2.0],
    'n_estimators': [200, 300],
}
clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=8,
    cv=_sgkf(),
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=0,
)
search.fit(X_train_sel, y_train)
print('best params', search.best_params_)
print('best cv auc', search.best_score_)
best = search.best_estimator_
y_prob = best.predict_proba(X_test_sel)[:, 1]
print('test acc', accuracy_score(y_test, y_prob))
print('test bal acc', balanced_accuracy_score(y_test, y_prob >= 0.5))
print('test auc', roc_auc_score(y_test, y_prob))
print('test positive rate', (y_prob >= 0.5).mean())
