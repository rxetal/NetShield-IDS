import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix)

# ==========================================
# PREPROCESSING & ENCODING (LEAK-FREE)
# ==========================================
def fit_and_save_categorical_encoder(df_train, cat_cols=['proto', 'service', 'state'], save_path="models/categorical_encoder.joblib"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(df_train[cat_cols])
    joblib.dump(encoder, save_path)
    return encoder

def transform_categorical_features(df, encoder, cat_cols=['proto', 'service', 'state']):
    df_transformed = df.copy()
    df_transformed[cat_cols] = encoder.transform(df_transformed[cat_cols])
    return df_transformed

# ==========================================
# TIER 1 MODEL FUNCTIONS
# ==========================================
def prepare_tier1_data(df, encoder):
    df_encoded = transform_categorical_features(df, encoder)
    X = df_encoded.drop(columns=['label', 'attack_cat'], errors='ignore')
    y = df_encoded['label']
    return X, y

def train_tier1_model(X_train, y_train):
    print("[TRAINING] Training Tier 1 XGBoost Binary Classifier...")
    model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        random_state=42, n_jobs=-1, tree_method='hist', eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_and_plot_tier1(model, X_test, y_test, output_dir="results"):
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    metrics_str = f"TIER 1 METRICS\nAcc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n"
    print(metrics_str)
    
    with open(os.path.join(output_dir, "metrics/tier1_report.txt"), "w") as f:
        f.write(metrics_str + "\n" + classification_report(y_test, y_pred, zero_division=0))
        
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Tier 1: Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures/tier1_confusion_matrix.png"), dpi=300)
    plt.close()

# ==========================================
# TIER 2 MODEL FUNCTIONS
# ==========================================
def prepare_tier2_data(df, feature_encoder, label_encoder=None):
    attack_df = df[df['label'] == 1].copy()
    attack_df = attack_df[attack_df['attack_cat'] != 'Normal']
    
    df_encoded = transform_categorical_features(attack_df, feature_encoder)
    X = df_encoded.drop(columns=['label', 'attack_cat'], errors='ignore')
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df_encoded['attack_cat'])
    else:
        y = label_encoder.transform(df_encoded['attack_cat'])
        
    return X, y, label_encoder

def train_tier2_model(X_train, y_train):
    print("[TRAINING] Training Tier 2 XGBoost Multiclass Classifier...")
    model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        random_state=42, n_jobs=-1, tree_method='hist',
        objective='multi:softprob', eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_and_plot_tier2(model, X_test, y_test, label_encoder, output_dir="results"):
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    
    y_pred = model.predict(X_test)
    classes = label_encoder.classes_
    labels = np.arange(len(classes))
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    metrics_str = f"TIER 2 METRICS\nAccuracy: {acc:.4f} | Macro F1: {f1_macro:.4f}\n"
    print(metrics_str)
    
    report = classification_report(
        y_test, y_pred, 
        labels=labels, 
        target_names=classes, 
        zero_division=0
    )
    
    with open(os.path.join(output_dir, "metrics/tier2_report.txt"), "w") as f:
        f.write(metrics_str + "\n" + report)
        
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=classes, yticklabels=classes)
    plt.title('Tier 2: Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figures/tier2_confusion_matrix.png"), dpi=300)
    plt.close()