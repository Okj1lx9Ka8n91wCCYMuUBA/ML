import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier

train_df = pd.read_csv('datasets/unimelb/unimelb_training.csv', low_memory=False)
test_df = pd.read_csv('datasets/unimelb/unimelb_test.csv', low_memory=False)

X = train_df.drop(columns=['Grant.Status'])
y = train_df['Grant.Status']

numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if X[col].notna().any()]

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if X[col].notna().any()]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)

X_train_processed = pd.DataFrame(X_train_processed, columns=numeric_cols + categorical_cols)
X_val_processed = pd.DataFrame(X_val_processed, columns=numeric_cols + categorical_cols)

for col in categorical_cols:
    X_train_processed[col] = X_train_processed[col].astype(str)
    X_val_processed[col] = X_val_processed[col].astype(str)

categorical_features_indices = [X_train_processed.columns.get_loc(col) for col in categorical_cols]

catboost_model = CatBoostClassifier(
    random_state=42,
    eval_metric='Logloss',
    early_stopping_rounds=10,
    verbose=0
)

catboost_model.fit(
    X_train_processed, y_train,
    eval_set=(X_val_processed, y_val),
    cat_features=categorical_features_indices,
    use_best_model=True
)

evals_result = catboost_model.evals_result_
train_logloss = evals_result['learn']['Logloss']
val_logloss = evals_result['validation']['Logloss']
epochs = range(len(train_logloss))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_logloss, label='Train')
plt.plot(epochs, val_logloss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.title('CatBoost Log Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve.png')
plt.close()

y_val_pred = catboost_model.predict(X_val_processed)
y_val_prob = catboost_model.predict_proba(X_val_processed)[:, 1]

report = classification_report(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_prob)

with open('metrics.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"\nROC AUC Score: {roc_auc:.4f}\n")

joblib.dump(preprocessor, 'preprocessor.joblib')
catboost_model.save_model('catboost_model.cbm')

param_grid = {
    'iterations': [100, 200],
    'depth': [4, 6],
    'learning_rate': [0.01, 0.1]
}

catboost_model_cv = CatBoostClassifier(
    random_state=42,
    eval_metric='Logloss',
    early_stopping_rounds=10,
    verbose=0
)

grid_search = GridSearchCV(
    estimator=catboost_model_cv,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=1
)

grid_search.fit(
    X_train_processed, y_train,
    eval_set=(X_val_processed, y_val),
    cat_features=categorical_features_indices,
    use_best_model=True
)

best_model = grid_search.best_estimator_
missing_cols = set(X.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = np.nan  
X_test = test_df[X.columns]

X_test_processed = preprocessor.transform(X_test)
X_test_processed = pd.DataFrame(X_test_processed, columns=numeric_cols + categorical_cols)

for col in categorical_cols:
    X_test_processed[col] = X_test_processed[col].astype(str)

test_pred = best_model.predict(X_test_processed)
test_prob = best_model.predict_proba(X_test_processed)[:, 1]

predictions = pd.DataFrame({
    'ID': test_df.index,
    'Grant.Status': test_pred,
    'Probability': test_prob
})

predictions.to_csv('predictions.csv', index=False)
