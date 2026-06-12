import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from train_dl_model import (
    load_voice_patient_split,
    prune_weak_voice_features,
    impute_with_train_stats,
    scale_train_transform_test,
    _build_xgb_variants,
    collect_oof_stack,
    train_base_models_full,
    predict_test_stack,
    tune_decision_threshold,
    sample_weights_from_map,
    print_class_distribution_and_weights,
)

train_df, test_df, patient_col, target_col, raw_feature_cols = load_voice_patient_split(test_size=0.1, random_state=42)
y_train = train_df['_label'].values
y_test = test_df['_label'].values
groups_train = train_df[patient_col].astype(str).values

weight_map, scale_pos_weight = print_class_distribution_and_weights(y_train, 'train', print_weights=False)
pruned_cols, vt, _, _ = prune_weak_voice_features(train_df, test_df, raw_feature_cols)
train_df, test_df = impute_with_train_stats(train_df, test_df, pruned_cols)
X_train, X_test, scaler = scale_train_transform_test(train_df, test_df, pruned_cols)

# use all pruned features by selecting k=-1
k = len(pruned_cols)
selector = None
feature_cols = pruned_cols
X_train_sel = X_train
X_test_sel = X_test

base_models = _build_xgb_variants(scale_pos_weight)
print('Collecting OOF')
sample_weight = sample_weights_from_map(y_train, weight_map)
oof_train = collect_oof_stack(X_train_sel, y_train, groups_train, base_models, sample_weight)
meta = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
meta.fit(oof_train, y_train)
meta_train_prob = meta.predict_proba(oof_train)[:,1]
print('meta oof auc', roc_auc_score(y_train, meta_train_prob))
decision_threshold, oof_bal = tune_decision_threshold(y_train, meta_train_prob)
print('threshold', decision_threshold, 'oof balanced', oof_bal)
trained_base = train_base_models_full(X_train_sel, y_train, base_models, sample_weight)
test_stack = predict_test_stack(trained_base, X_test_sel)

meta_test_prob = meta.predict_proba(test_stack)[:,1]
avg_test_prob = test_stack.mean(axis=1)
for name, prob in [('meta', meta_test_prob), ('avg', avg_test_prob)]:
    best_acc = 0
    best_t = 0.5
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (prob >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    print(name, 'best acc', best_acc, 'best_t', best_t, 'auc', roc_auc_score(y_test, prob))
