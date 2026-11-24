"""
Machine Learning & Pacing Models for Short-Form Video Retention and Virality.
Trains:
1. Completion Rate Regressor (Random Forest Regressor)
2. Viral Probability Classifier (Random Forest Classifier)
Computes Feature Importances to explain the exact levers of short-form retention.
"""
import os
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "viral_videos.duckdb")
MODEL_PATH = os.path.join(BASE_DIR, "models", "viral_models.joblib")

def train_retention_models():
    print("🧠 Treinando modelos preditivos de retenção e probabilidade viral...")
    con = duckdb.connect(DB_PATH)
    df = con.execute("SELECT * FROM videos").df()
    con.close()

    # One-hot encode categorical features: hook_type, platform, niche
    features_num = ["duration_sec", "wpm", "cuts_per_minute", "hook_duration_sec", "has_trending_audio", "has_dynamic_subtitles"]
    
    df_encoded = pd.get_dummies(df[["platform", "niche", "hook_type"]], drop_first=False)
    X = pd.concat([df[features_num], df_encoded], axis=1)
    
    y_completion = df["completion_rate_pct"]
    y_viral = df["is_viral"]

    X_train, X_test, y_comp_train, y_comp_test, y_vir_train, y_vir_test = train_test_split(
        X, y_completion, y_viral, test_size=0.2, random_state=42
    )

    # 1. Regressor de Taxa de Conclusão
    reg = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_comp_train)
    y_comp_pred = reg.predict(X_test)
    r2 = r2_score(y_comp_test, y_comp_pred)
    mae = mean_absolute_error(y_comp_test, y_comp_pred)
    print(f"🎯 Regressor de Conclusão -> R² Score: {r2:.4f} | MAE: {mae:.2f}%")

    # 2. Classificador de Potencial Viral
    clf = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_vir_train)
    y_vir_prob = clf.predict_proba(X_test)[:, 1]
    y_vir_pred = clf.predict(X_test)
    auc = roc_auc_score(y_vir_test, y_vir_prob)
    acc = accuracy_score(y_vir_test, y_vir_pred)
    print(f"🔥 Classificador Viral -> ROC-AUC: {auc:.4f} | Acurácia: {acc*100:.2f}%")

    # Feature Importance Top 8
    importances = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(8).to_dict()
    print("📈 Top 5 Fatores Determinantes de Retenção:")
    for feat, imp in list(top_features.items())[:5]:
        print(f"   • {feat}: {imp*100:.1f}%")

    bundle = {
        "regressor": reg,
        "classifier": clf,
        "feature_names": list(X.columns),
        "categorical_columns": list(df_encoded.columns),
        "top_features": top_features,
        "metrics": {
            "r2_score": round(float(r2), 4),
            "mae": round(float(mae), 2),
            "roc_auc": round(float(auc), 4),
            "accuracy": round(float(acc * 100), 2)
        }
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"✅ Modelos salvos com sucesso em: {MODEL_PATH}")
    return bundle

if __name__ == "__main__":
    train_retention_models()
