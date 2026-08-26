import pandas as pd
import joblib
from src.models import prepare_tier1_data, train_tier1_model, evaluate_and_plot_tier1

print("==================================================")
print("             RUNNING PHASE 5: TIER 1              ")
print("==================================================")

train_df = pd.read_parquet("data/processed/train_set.parquet")
test_df = pd.read_parquet("data/processed/test_set.parquet")
cat_encoder = joblib.load("models/categorical_encoder.joblib")

X_train, y_train = prepare_tier1_data(train_df, cat_encoder)
X_test, y_test = prepare_tier1_data(test_df, cat_encoder)

model_t1 = train_tier1_model(X_train, y_train)
evaluate_and_plot_tier1(model_t1, X_test, y_test)
joblib.dump(model_t1, "models/tier1_xgb.joblib")
print("✅ Phase 5 Completed Successfully.")
