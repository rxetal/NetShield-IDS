import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from src.models import (
    fit_and_save_categorical_encoder,
    prepare_tier1_data, train_tier1_model, evaluate_and_plot_tier1,
    prepare_tier2_data, train_tier2_model, evaluate_and_plot_tier2
)

print("==================================================")
print("     NETSHIELD CANONICAL LEAK-FREE PIPELINE       ")
print("==================================================")

dataset_paths = [
    'data/processed/cleaned_unsw_nb15.parquet',
    'data/processed/clean_unsw_nb15.parquet'
]
data_path = next((p for p in dataset_paths if os.path.exists(p)), None)
if not data_path:
    raise FileNotFoundError("🚨 Cleaned dataset missing in data/processed/")

df = pd.read_parquet(data_path).drop_duplicates().reset_index(drop=True)

# Master Train/Test Split with Attack Category Stratification
print("\n[STEP 1] Performing Master Train/Test Split (80/20 with Attack-Category Stratification)...")
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['attack_cat']
)

os.makedirs("data/processed", exist_ok=True)
train_df.to_parquet("data/processed/train_set.parquet")
test_df.to_parquet("data/processed/test_set.parquet")

# Fit on Train Set only
print("\n[STEP 2] Fitting Categorical Encoder on TRAIN SET ONLY...")
cat_cols = ['proto', 'service', 'state']
cat_encoder = fit_and_save_categorical_encoder(train_df, cat_cols=cat_cols)

# Tier 1 Execution
print("\n[STEP 3] Preparing & Training Tier 1 Classifier...")
X_train_t1, y_train_t1 = prepare_tier1_data(train_df, cat_encoder)
X_test_t1, y_test_t1 = prepare_tier1_data(test_df, cat_encoder)

tier1_model = train_tier1_model(X_train_t1, y_train_t1)
evaluate_and_plot_tier1(tier1_model, X_test_t1, y_test_t1)
joblib.dump(tier1_model, "models/tier1_xgb.joblib")

# Tier 2 Execution
print("\n[STEP 4] Preparing & Training Tier 2 Classifier...")
X_train_t2, y_train_t2, label_encoder = prepare_tier2_data(train_df, cat_encoder)
X_test_t2, y_test_t2, _ = prepare_tier2_data(test_df, cat_encoder, label_encoder=label_encoder)

joblib.dump(label_encoder, "models/label_encoder.joblib")

tier2_model = train_tier2_model(X_train_t2, y_train_t2)
evaluate_and_plot_tier2(tier2_model, X_test_t2, y_test_t2, label_encoder)
joblib.dump(tier2_model, "models/tier2_xgb.joblib")

print("\n==================================================")
print("     SUCCESS: LEAK-FREE PIPELINE COMPLETED        ")
print("==================================================")