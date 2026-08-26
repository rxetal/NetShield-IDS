import pandas as pd
import joblib
from src.models import prepare_tier2_data, train_tier2_model, evaluate_and_plot_tier2

print("==================================================")
print("             RUNNING PHASE 6: TIER 2              ")
print("==================================================")

train_df = pd.read_parquet("data/processed/train_set.parquet")
test_df = pd.read_parquet("data/processed/test_set.parquet")
cat_encoder = joblib.load("models/categorical_encoder.joblib")

X_train, y_train, label_encoder = prepare_tier2_data(train_df, cat_encoder)
X_test, y_test, _ = prepare_tier2_data(test_df, cat_encoder, label_encoder=label_encoder)

model_t2 = train_tier2_model(X_train, y_train)
evaluate_and_plot_tier2(model_t2, X_test, y_test, label_encoder)
joblib.dump(model_t2, "models/tier2_xgb.joblib")
print("✅ Phase 6 Completed Successfully.")
