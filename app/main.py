"""
Viral Analytics - Short-Form Creator & Algorithmic Retention Intelligence (FastAPI)
Predicts completion rates, WPM speech pacing and viral probability for TikTok, Reels, and Shorts.
"""
import os
import duckdb
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "viral_videos.duckdb")
MODEL_PATH = os.path.join(BASE_DIR, "models", "viral_models.joblib")
TEMPLATES_DIR = os.path.join(BASE_DIR, "app", "templates")

app = FastAPI(title="Viral Analytics - Short-Form Creator Intelligence", version="1.0.0")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Carrega os modelos de Machine Learning
model_bundle = joblib.load(MODEL_PATH)
reg_completion = model_bundle["regressor"]
clf_viral = model_bundle["classifier"]
feature_names = model_bundle["feature_names"]
metrics = model_bundle["metrics"]

def get_db():
    return duckdb.connect(DB_PATH, read_only=True)

class VideoSimulationRequest(BaseModel):
    platform: str
    niche: str
    hook_type: str
    duration_sec: int
    wpm: int
    cuts_per_minute: float
    hook_duration_sec: float
    has_trending_audio: bool
    has_dynamic_subtitles: bool

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "metrics": metrics})

@app.get("/api/summary")
async def get_summary():
    con = get_db()
    total_vids = con.execute("SELECT count(*) FROM videos").fetchone()[0]
    total_views = con.execute("SELECT sum(views) FROM videos").fetchone()[0]
    avg_wpm = con.execute("SELECT round(avg(wpm), 1) FROM videos").fetchone()[0]
    avg_completion = con.execute("SELECT round(avg(completion_rate_pct), 1) FROM videos").fetchone()[0]
    viral_rate = con.execute("SELECT round(sum(is_viral)*100.0/count(*), 1) FROM videos").fetchone()[0]
    con.close()

    return {
        "total_videos": total_vids,
        "total_views": total_views,
        "avg_wpm": avg_wpm,
        "avg_completion": avg_completion,
        "viral_rate": viral_rate
    }

@app.get("/api/hook_performance")
async def get_hook_performance():
    con = get_db()
    rows = con.execute("""
        SELECT 
            hook_type,
            count(*) as total,
            round(avg(retention_3s_pct), 1) as avg_ret_3s,
            round(avg(completion_rate_pct), 1) as avg_completion,
            round(sum(is_viral)*100.0/count(*), 1) as viral_pct,
            round(avg(views), 0) as avg_views
        FROM videos
        GROUP BY hook_type
        ORDER BY viral_pct DESC
    """).fetchall()
    con.close()

    return [
        {
            "hook_type": r[0].split("(")[0].strip(),
            "full_hook": r[0],
            "total": r[1],
            "avg_ret_3s": float(r[2]),
            "avg_completion": float(r[3]),
            "viral_pct": float(r[4]),
            "avg_views": int(r[5])
        } for r in rows
    ]

@app.get("/api/wpm_distribution")
async def get_wpm_distribution():
    con = get_db()
    rows = con.execute("""
        SELECT 
            CAST(wpm / 15 * 15 AS INT) as wpm_bucket,
            count(*) as count,
            round(avg(completion_rate_pct), 1) as avg_completion,
            round(sum(is_viral)*100.0/count(*), 1) as viral_rate
        FROM videos
        GROUP BY wpm_bucket
        ORDER BY wpm_bucket ASC
    """).fetchall()
    con.close()

    return [
        {
            "bucket": f"{r[0]}-{r[0]+15} WPM",
            "count": r[1],
            "avg_completion": float(r[2]),
            "viral_rate": float(r[3])
        } for r in rows
    ]

@app.get("/api/platform_metrics")
async def get_platform_metrics():
    con = get_db()
    rows = con.execute("""
        SELECT 
            platform,
            count(*) as total,
            sum(views) as total_views,
            round(avg(retention_3s_pct), 1) as ret_3s,
            round(avg(completion_rate_pct), 1) as completion,
            round(sum(is_viral)*100.0/count(*), 1) as viral_pct
        FROM videos
        GROUP BY platform
        ORDER BY total_views DESC
    """).fetchall()
    con.close()

    return [
        {
            "platform": r[0],
            "total": r[1],
            "views": int(r[2]),
            "ret_3s": float(r[3]),
            "completion": float(r[4]),
            "viral_pct": float(r[5])
        } for r in rows
    ]

@app.post("/api/predict_video")
async def predict_video(req: VideoSimulationRequest):
    # Constrói o vetor de features compatível com o encoder
    input_dict = {
        "duration_sec": req.duration_sec,
        "wpm": req.wpm,
        "cuts_per_minute": req.cuts_per_minute,
        "hook_duration_sec": req.hook_duration_sec,
        "has_trending_audio": int(req.has_trending_audio),
        "has_dynamic_subtitles": int(req.has_dynamic_subtitles),
    }

    # Preenche as colunas dummy
    for col in feature_names:
        if col not in input_dict:
            input_dict[col] = 0

    col_platform = f"platform_{req.platform}"
    col_niche = f"niche_{req.niche}"
    col_hook = f"hook_type_{req.hook_type}"

    if col_platform in input_dict:
        input_dict[col_platform] = 1
    if col_niche in input_dict:
        input_dict[col_niche] = 1
    if col_hook in input_dict:
        input_dict[col_hook] = 1

    df_sample = pd.DataFrame([input_dict])[feature_names]

    # Previsão
    pred_completion = float(reg_completion.predict(df_sample)[0])
    pred_viral_prob = float(clf_viral.predict_proba(df_sample)[0][1])

    # Dicas algorítmicas acionáveis
    tips = []
    if req.wpm < 140:
        tips.append("⚠️ Ritmo de fala lento (< 140 WPM). Aumente a cadência ou corte pausas/respirações para reter a audiência.")
    elif req.wpm > 215:
        tips.append("⚠️ Ritmo acelerado demais (> 215 WPM). Pode prejudicar a clareza e causar rejeição no meio do vídeo.")
    else:
        tips.append("✅ Cadência perfeita (150-195 WPM)! Ritmo ideal para retenção máxima.")

    if req.cuts_per_minute < 18:
        tips.append("✂️ Adicione mais zoom-ins e cortes de câmera (recomendado: 20-30 cortes por minuto).")
    else:
        tips.append("✅ Excelente dinamismo visual e alternância de planos.")

    if "Direct Statement" in req.hook_type:
        tips.append("🎣 Alterne para 'Curiosity Gap' ou 'Negative Warning' nos primeiros 3s para dobrar a taxa de visualização de 3s.")

    return {
        "completion_rate_pct": round(pred_completion, 1),
        "viral_probability_pct": round(pred_viral_prob * 100, 1),
        "classification": "🔥 ALTO POTENCIAL VIRAL (>250k views)" if pred_viral_prob > 0.60 else ("⚡ DISTRIBUIÇÃO MÉDIA" if pred_viral_prob > 0.35 else "📉 ALCANCE LIMITADO"),
        "estimated_retention_3s": round(min(95.0, pred_completion * 1.35), 1),
        "actionable_tips": tips
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8005, reload=True)
