import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBClassifierWithEval(XGBClassifier, BaseEstimator, ClassifierMixin):
    def fit(self, X, y, **kwargs):
        eval_set = kwargs.pop('eval_set', None)
        super().fit(X, y, eval_set=eval_set, **kwargs)
        return self

train_df = pd.read_csv('datasets/unimelb/unimelb_training.csv', low_memory=False)
test_df = pd.read_csv('datasets/unimelb/unimelb_test.csv', low_memory=False)

X = train_df.drop(columns=['Grant.Status'])
y = train_df['Grant.Status']

numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
numeric_cols = [col for col in numeric_cols if X[col].notna().any()]

categorical_cols = X.select_dtypes(include=['object']).columns
categorical_cols = [col for col in categorical_cols if X[col].notna().any()]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifierWithEval(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=10
    ))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)

xgb_model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=10
)

eval_set = [(X_train_processed, y_train), (X_val_processed, y_val)]

xgb_model.fit(
    X_train_processed,
    y_train,
    eval_set=eval_set,
    verbose=True
)

results = xgb_model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve.png')
plt.close()

y_val_pred = xgb_model.predict(X_val_processed)
y_val_prob = xgb_model.predict_proba(X_val_processed)[:, 1]

report = classification_report(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_prob)

with open('metrics.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"\nROC AUC Score: {roc_auc:.4f}\n")

joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(xgb_model, 'xgb_model.joblib')

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

test_numeric = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(test_df[numeric_cols]), columns=numeric_cols)
test_categorical = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(test_df[categorical_cols]), columns=categorical_cols)
test_categorical = pd.DataFrame(OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(test_categorical), columns=categorical_cols)
test_df_processed = pd.concat([test_numeric, test_categorical], axis=1)
test_df_processed = pd.DataFrame(StandardScaler().fit_transform(test_df_processed), columns=test_df_processed.columns)

test_pred = best_model['classifier'].predict(test_df_processed)
test_prob = best_model['classifier'].predict_proba(test_df_processed)[:, 1]

predictions = pd.DataFrame({
    'ID': test_df.index,
    'Grant.Status': test_pred,
    'Probability': test_prob
})

predictions.to_csv('predictions.csv', index=False)
