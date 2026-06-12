import numpy as np
import joblib
from train_dl_model import load_voice_patient_split, prune_weak_voice_features, impute_with_train_stats, scale_train_transform_test, select_k_best_features
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

train_df, test_df, patient_col, target_col, raw_feature_cols = load_voice_patient_split(test_size=0.1, random_state=42)
y_train = train_df['_label'].values
y_test = test_df['_label'].values
pruned_cols, vt, _, _ = prune_weak_voice_features(train_df, test_df, raw_feature_cols)
train_df, test_df = impute_with_train_stats(train_df, test_df, pruned_cols)
X_train, X_test, scaler = scale_train_transform_test(train_df, test_df, pruned_cols)
# Load feature selector and selected features
selector = joblib.load('models/feature_selector.pkl')
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)
base_models = joblib.load('models/voice_xgb_base_models.pkl')
meta = joblib.load('models/voice_meta_learner.pkl')

train_stack = np.column_stack([m.predict_proba(X_train_sel)[:,1] for m in base_models])
test_stack = np.column_stack([m.predict_proba(X_test_sel)[:,1] for m in base_models])
avg_train_prob = train_stack.mean(axis=1)
avg_test_prob = test_stack.mean(axis=1)
meta_train_prob = meta.predict_proba(train_stack)[:,1]
meta_test_prob = meta.predict_proba(test_stack)[:,1]

for name, prob in [('avg', avg_test_prob), ('meta', meta_test_prob)]:
    pred = (prob >= 0.5).astype(int)
    print(name, accuracy_score(y_test, pred), balanced_accuracy_score(y_test, pred), roc_auc_score(y_test, prob), precision_score(y_test, pred), recall_score(y_test, pred))
