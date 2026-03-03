import os
import sys
import joblib
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ==============================
# Logging Configuration
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==============================
# Constants
# ==============================

DATA_PATH = "data/application_train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "credit_risk_model.pkl")

# ==============================
# Load Data
# ==============================

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.error(f"Dataset not found at {path}")
        sys.exit(1)

    logger.info("Loading dataset...")
    df = pd.read_csv(path)
    logger.info(f"Dataset shape: {df.shape}")
    return df


# ==============================
# Clean Data
# ==============================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Dropping high-missing columns (>60%)")
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.6].index
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Shape after cleaning: {df.shape}")
    return df


# ==============================
# Build Pipeline
# ==============================

def build_pipeline(X: pd.DataFrame) -> Pipeline:

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline


# ==============================
# Train Model
# ==============================

def train():

    df = load_data(DATA_PATH)
    df = clean_data(df)

    if "TARGET" not in df.columns:
        logger.error("TARGET column missing!")
        sys.exit(1)

    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]

    pipeline = build_pipeline(X)

    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model...")
    preds = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    logger.info(f"Final ROC-AUC: {round(auc, 4)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    metadata = {
        "model": pipeline,
        "trained_at": datetime.now().isoformat(),
        "roc_auc": float(auc),
        "feature_count": X.shape[1]
    }

    joblib.dump(metadata, MODEL_PATH)

    logger.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()

    