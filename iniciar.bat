@echo off
title Viral Analytics - Short-Form Creator Intelligence
color 0D
echo ========================================================
echo   VIRAL ANALYTICS - RETENCAO & PACING DE VIDEOS CURTOS
echo   TikTok / YouTube Shorts / Instagram Reels
echo   DuckDB + Random Forest + FastAPI
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/3] Verificando dependencias Python...
py -3.13 -m pip install -q -r requirements.txt

echo.
echo [2/3] Verificando banco DuckDB e modelos de Machine Learning...
if not exist "data\viral_videos.duckdb" (
    echo Gerando dataset analitico de videos curtos...
    py -3.13 data\pipeline.py
)

if not exist "models\viral_models.joblib" (
    echo Treinando modelos preditivos de retencao...
    py -3.13 models\retention_predictor.py
)

echo.
echo [3/3] Iniciando Servidor Web na porta 8005...
echo Acesse no seu navegador: http://localhost:8005
echo.
py -3.13 -m uvicorn app.main:app --host 127.0.0.1 --port 8005 --reload

pause
