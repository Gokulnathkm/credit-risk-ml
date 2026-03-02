import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("Loading dataset...")

df = pd.read_csv("data/application_train.csv")

print("Original Shape:", df.shape)

# -----------------------
# 1. Drop ID column
# -----------------------
df = df.drop(columns=["SK_ID_CURR"])

# -----------------------
# 2. Separate target
# -----------------------
X = df.drop("TARGET", axis=1)
y = df["TARGET"]

# -----------------------
# 3. Remove high-missing columns (>40%)
# -----------------------
missing_ratio = X.isnull().mean()
high_missing_cols = missing_ratio[missing_ratio > 0.4].index
X = X.drop(columns=high_missing_cols)

print("Shape after dropping high-missing columns:", X.shape)

# -----------------------
# 4. Identify feature types
# -----------------------
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

# -----------------------
# 5. Preprocessing Pipelines
# -----------------------
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features)
])

# -----------------------
# 6. Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# -----------------------
# 7. Define Models
# -----------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

results = {}

for name, clf in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, y_pred_proba)

    results[name] = roc_score

    print(f"{name} ROC-AUC: {roc_score:.4f}")

print("\nFinal Model Comparison:")
for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}")

    