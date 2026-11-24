# 📱 Short-Form Creator & Viral Video Analytics

> **Inteligência preditiva de retenção, velocidade de fala (WPM), ganchos nos primeiros 3 segundos e probabilidade viral em vídeos curtos (TikTok, YouTube Shorts e Instagram Reels).**

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-10k_Videos-yellow.svg)
![Scikit-Learn](https://img.shields.io/badge/ML-Random_Forest-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Dashboard-009688.svg)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-Dark_Mode-38bdf8.svg)

---

## 🎯 Visão Geral do Projeto

Os algoritmos de distribuição em massa de vídeos curtos priorizam duas métricas fundamentais: a **taxa de retenção nos primeiros 3 segundos (Drop-off Inicial)** e a **taxa de conclusão do vídeo (Completion Rate)**.

O **Viral Analytics** processa e analisa 10.000 publicações em alta performance para revelar exatamente o impacto de cadência verbal (palavras por minuto), tipos de ganchos psicológicos e dinamismo de cortes na retenção do espectador.

### 🔍 Principais Funcionalidades
1. **Modelagem de Retenção & Pacing:** Correlação estatística entre WPM (110 a 240 palavras/minuto) e conclusão do vídeo, identificando o *sweet spot* entre 165 e 195 WPM.
2. **Taxonomia dos Primeiros 3 Segundos:** Comparativo de eficácia entre *Curiosity Gap*, *Negative Warning*, *Visual Disruption*, *Question Hook* e *Direct Statement*.
3. **Machine Learning Duplo (Random Forest):**
   - **Regressor de Taxa de Conclusão:** R² = 0.8081, MAE = 5.17%.
   - **Classificador de Probabilidade Viral:** ROC-AUC = 0.9648, Acurácia = 89.60%.
4. **Simulador em Tempo Real:** Ferramenta interativa onde criadores ajustam duração, WPM, cortes e ganchos para testar a probabilidade algorítmica antes de postar, acompanhada de sugestões práticas de edição.
5. **Dashboard Moderno (Sem Streamlit):** Desenvolvido em FastAPI, DuckDB, Tailwind CSS Dark Mode e Chart.js.

---

## 🚀 Como Executar no Windows (1-Clique)

Basta dar dois cliques no arquivo:
```cmd
iniciar.bat
```
Ou pelo terminal:
```bash
# Instalar dependências
py -3.13 -m pip install -r requirements.txt

# Iniciar dashboard interativo
py -3.13 -m uvicorn app.main:app --host 127.0.0.1 --port 8005 --reload
```
Acesse no seu navegador: **http://localhost:8005**

---

## 📊 Estrutura do Repositório

```
shortform-viral-analytics/
├── data/
│   ├── pipeline.py            # Ingestão e modelagem de retenção algorítmica
│   └── viral_videos.duckdb    # Base analítica DuckDB (10.000 vídeos)
├── models/
│   ├── retention_predictor.py # Treinamento do regressor e classificador Random Forest
│   └── viral_models.joblib    # Modelos e encoders serializados
├── app/
│   ├── main.py                # API RESTful FastAPI
│   └── templates/
│       └── index.html         # Frontend Tailwind CSS + Chart.js com Simulador
├── iniciar.bat                # Executável Windows de 1 clique
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Autor

Desenvolvido por **[Tassio Sales](https://github.com/TassioSales)**  
Engenharia de Dados & Modelagem Preditiva.
