import os
import joblib
import pandas as pd
import numpy as np

class NetShieldPipeline:
    def __init__(self, tier1_path="models/tier1_xgboost.joblib", 
                 tier2_path="models/tier2_xgboost.joblib", 
                 label_encoder_path="models/label_encoder.joblib",
                 cat_encoder_path="models/categorical_encoder.joblib"):
        
        self.tier1_model = joblib.load(tier1_path)
        self.tier2_model = joblib.load(tier2_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.cat_encoder = joblib.load(cat_encoder_path)
        self.cat_cols = ['proto', 'service', 'state']

    def predict_single(self, sample_df):
        df_processed = sample_df.copy()
        
        # تحويل البيانات باستخدام القاموس الموحد
        df_processed[self.cat_cols] = self.cat_encoder.transform(df_processed[self.cat_cols])
        X = df_processed.drop(columns=['label', 'attack_cat'], errors='ignore')
        
        tier1_pred = self.tier1_model.predict(X)[0]
        tier1_prob = self.tier1_model.predict_proba(X)[0][1]
        
        # حساب نسبة الخطورة
        risk_score = round(float(tier1_prob) * 100, 2)
        
        if tier1_pred == 0:
            return {
                "is_attack": False,
                "tier1_label": "Normal",
                "attack_prob": float(tier1_prob),
                "attack_type": "Normal",
                "risk_score": risk_score,
                "severity": "Low"
            }
        else:
            tier2_pred_idx = self.tier2_model.predict(X)[0]
            attack_name = self.label_encoder.inverse_transform([tier2_pred_idx])[0]
            severity = "Critical" if risk_score > 85 else ("High" if risk_score > 60 else "Medium")
            
            return {
                "is_attack": True,
                "tier1_label": "Attack",
                "attack_prob": float(tier1_prob),
                "attack_type": attack_name,
                "risk_score": risk_score,
                "severity": severity
            }

    def predict(self, sample_df):
        df_processed = sample_df.copy()
        
        # تحويل الأعمدة النصية باستخدام القاموس
        df_processed[self.cat_cols] = self.cat_encoder.transform(df_processed[self.cat_cols])
        X = df_processed.drop(columns=['label', 'attack_cat'], errors='ignore')
        
        # التوقع من Tier 1
        t1_preds = self.tier1_model.predict(X)
        t1_probas = self.tier1_model.predict_proba(X)[:, 1]
        
        # التوقع النهائي (Tier 2 لو فيه هجوم)
        final_preds = []
        for i, pred in enumerate(t1_preds):
            if pred == 0:
                final_preds.append("Normal")
            else:
                tier2_pred_idx = self.tier2_model.predict(X.iloc[[i]])[0]
                attack_name = self.label_encoder.inverse_transform([tier2_pred_idx])[0]
                final_preds.append(attack_name)
                
        return final_preds, t1_preds, t1_probas