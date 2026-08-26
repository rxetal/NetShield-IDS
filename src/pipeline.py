import os
import joblib
import pandas as pd
import numpy as np
from src.models import transform_categorical_features

class NetShieldPipeline:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        
        # تحميل الـ Artifacts المحفوظة
        self.cat_encoder = joblib.load(os.path.join(models_dir, "categorical_encoder.joblib"))
        self.tier1_model = joblib.load(os.path.join(models_dir, "tier1_xgb.joblib"))
        self.tier2_model = joblib.load(os.path.join(models_dir, "tier2_xgb.joblib"))
        self.label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))

    def _prepare_features(self, df_raw):
        """دالة مساعدة لتحويل الخصائص وتأكيد ترتيب الأعمدة طبقاً للنموذج"""
        # 1. تطبيق الـ Categorical Encoding الموحد
        df_encoded = transform_categorical_features(df_raw, self.cat_encoder)
        
        # 2. حذف أعمدة الـ Targets إن وجدت
        X = df_encoded.drop(columns=['label', 'attack_cat'], errors='ignore')
        
        # 3. إعادة ترتيب الأعمدة لتطابق ترتيب التدريب الخاص بـ XGBoost
        expected_features = getattr(self.tier1_model, "feature_names_in_", None)
        if expected_features is not None:
            X = X[expected_features]
            
        return X

    def predict_single(self, df_raw):
        """معالجة وتوقع حزمة شبكية واحدة بأسلوب الهرمي (Hierarchical)"""
        # تجهيز الخصائص
        X = self._prepare_features(df_raw)
        
        # 1. Tier 1 Prediction
        t1_pred = int(self.tier1_model.predict(X)[0])
        t1_proba = float(self.tier1_model.predict_proba(X)[0][1])
        
        if t1_pred == 0:
            return {
                "tier1_label": "Normal",
                "attack_type": "Normal",
                "attack_prob": t1_proba,
                "risk_score": round(t1_proba * 20, 2), # سكور منخفض للحركات الطبيعية
                "severity": "Low"
            }
        else:
            # 2. Tier 2 Prediction (في حالة كشف هجوم)
            t2_pred_idx = int(self.tier2_model.predict(X)[0])
            t2_probas = self.tier2_model.predict_proba(X)[0]
            
            attack_type = str(self.label_encoder.inverse_transform([t2_pred_idx])[0])
            max_t2_prob = float(np.max(t2_probas))
            
            # حساب الـ Risk Score و الدرجة
            risk_score = round(50 + (max_t2_prob * 50), 2)
            severity = "Critical" if risk_score > 85 else ("High" if risk_score > 70 else "Medium")
            
            return {
                "tier1_label": "Attack",
                "attack_type": attack_type,
                "attack_prob": max_t2_prob,
                "risk_score": risk_score,
                "severity": severity
            }

    def predict_batch(self, df_raw):
        """توقع كل البيانات دفعة واحدة بكتل موجهة (Fast Vectorized Inference)"""
        # تجهيز الخصائص لكل البيانات
        X = self._prepare_features(df_raw)
        
        # 1. Tier 1 Predictions دفعة واحدة
        t1_predictions = self.tier1_model.predict(X)
        y_pred_tier1 = np.asarray(t1_predictions, dtype=int)
        
        # 2. تهيئة مصفوفة التوقعات النهائية بقيم افتراضية "Normal"
        y_pred_final = np.full(len(X), "Normal", dtype=object)
        
        # 3. تحديد أماكن العينات التي تم تصنيفها كـ Attack
        attack_indices = np.where(y_pred_tier1 == 1)[0]
        
        # 4. تشغيل Tier 2 فقط على العينات المسجلة كـ Attack
        if len(attack_indices) > 0:
            X_attack = X.iloc[attack_indices]
            t2_predictions = self.tier2_model.predict(X_attack)
            t2_predictions = np.asarray(t2_predictions, dtype=int)
            
            # تحويل الأرقام إلى أسماء الهجمات النصية
            attack_types = self.label_encoder.inverse_transform(t2_predictions)
            y_pred_final[attack_indices] = attack_types
            
        return y_pred_tier1, y_pred_final