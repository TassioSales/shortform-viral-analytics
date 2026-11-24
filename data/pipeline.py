"""
Short-Form Creator & Viral Video Analytics Pipeline.
Ingests and generates 10,000 video records with retention curves, WPM speech pacing,
first 3-second hook taxonomy, and algorithmic distribution metrics.
Stores everything in DuckDB for sub-second analytical aggregations.
"""
import os
import duckdb
import numpy as np
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "viral_videos.duckdb")

PLATFORMS = ["TikTok", "YouTube Shorts", "Instagram Reels"]
NICHES = [
    ("Tech & Inteligência Artificial", 185, 0.22),
    ("Finanças & Investimentos", 165, 0.18),
    ("Curiosidades & Fatos Ocultos", 195, 0.25),
    ("True Crime & Mistério", 150, 0.12),
    ("Pets & Vlogs Humanizados", 160, 0.13),
    ("Humor & Reacts Rápidos", 210, 0.10)
]

HOOK_TYPES = [
    ("Curiosity Gap ('O segredo que ninguém te conta...')", 1.35),
    ("Negative Warning ('Pare de cometer esse erro agora...')", 1.45),
    ("Visual Disruption (Corte dinâmico / Objeto inesperado)", 1.25),
    ("Question Hook ('Você sabia que 90% das pessoas...?')", 1.10),
    ("Direct Statement ('Hoje vou te mostrar como...')", 0.85)
]

def generate_video_dataset(num_records=10000):
    np.random.seed(42)
    print("📱 Gerando base de dados analítica de vídeos curtos & retenção algorítmica...")

    niche_names = [n[0] for n in NICHES]
    niche_target_wpm = {n[0]: n[1] for n in NICHES}
    niche_weights = [n[2] for n in NICHES]

    hook_names = [h[0] for h in HOOK_TYPES]
    hook_boosts = {h[0]: h[1] for h in HOOK_TYPES}

    records = []

    for i in range(num_records):
        platform = np.random.choice(PLATFORMS, p=[0.45, 0.35, 0.20])
        niche = np.random.choice(niche_names, p=niche_weights)
        hook = np.random.choice(hook_names, p=[0.30, 0.28, 0.22, 0.12, 0.08])
        
        # Pacing & Speech
        base_wpm = niche_target_wpm[niche]
        wpm = int(np.clip(np.random.normal(base_wpm, 18), 110, 240))
        duration = int(np.random.choice([15, 25, 30, 45, 60, 75, 90], p=[0.10, 0.25, 0.30, 0.20, 0.08, 0.04, 0.03]))
        
        # Cortes por minuto (dinamismo)
        cuts_pm = round(float(np.clip(np.random.normal(24.0, 6.0), 10.0, 42.0)), 1)
        hook_duration = round(float(np.random.uniform(1.8, 3.8)), 1)
        has_trending_audio = int(np.random.rand() < 0.40)
        has_dynamic_subtitles = int(np.random.rand() < 0.82)
        
        # Algorithmic Retention Modeling
        hook_factor = hook_boosts[hook]
        wpm_penalty = 1.0 - abs(wpm - 180) / 300.0 # Sweet spot em torno de 180 WPM
        pacing_bonus = min(1.2, cuts_pm / 22.0)
        
        # Retenção nos primeiros 3 segundos (Drop-off inicial)
        base_ret_3s = 0.62 * hook_factor * pacing_bonus
        retention_3s = float(np.clip(base_ret_3s + np.random.normal(0, 0.06), 0.25, 0.96))
        
        # Retenção 50%
        retention_50pct = float(np.clip(retention_3s * (0.65 + 0.15 * wpm_penalty) + np.random.normal(0, 0.05), 0.15, 0.85))
        
        # Taxa de Conclusão (Completion Rate)
        duration_penalty = max(0.4, 1.0 - (duration / 120.0))
        completion_rate = float(np.clip(retention_50pct * duration_penalty * 1.1 + np.random.normal(0, 0.04), 0.08, 0.82))
        
        # Views & Virality Score
        virality_index = (retention_3s * 0.4) + (completion_rate * 0.5) + (has_trending_audio * 0.1)
        
        if virality_index > 0.62 and completion_rate > 0.48:
            # Algoritmo entrega para massa
            views = int(np.random.exponential(450000) + 180000)
            is_viral = 1
        else:
            views = int(np.random.exponential(18000) + 1200)
            is_viral = 0

        likes = int(views * np.random.uniform(0.06, 0.12))
        shares = int(views * (0.015 if is_viral else 0.003) * hook_factor)
        comments = int(views * np.random.uniform(0.002, 0.012))

        records.append({
            "video_id": f"VID-{100000 + i}",
            "platform": platform,
            "niche": niche,
            "duration_sec": duration,
            "wpm": wpm,
            "cuts_per_minute": cuts_pm,
            "hook_type": hook,
            "hook_duration_sec": hook_duration,
            "has_trending_audio": has_trending_audio,
            "has_dynamic_subtitles": has_dynamic_subtitles,
            "retention_3s_pct": round(retention_3s * 100, 1),
            "retention_50pct_pct": round(retention_50pct * 100, 1),
            "completion_rate_pct": round(completion_rate * 100, 1),
            "views": views,
            "likes": likes,
            "shares": shares,
            "comments": comments,
            "is_viral": is_viral,
            "virality_score": round(float(virality_index * 100), 1)
        })

    df = pd.DataFrame(records)

    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS videos")
    con.execute("CREATE TABLE videos AS SELECT * FROM df")
    total_vids = con.execute("SELECT count(*) FROM videos").fetchone()[0]
    total_views = con.execute("SELECT sum(views) FROM videos").fetchone()[0]
    total_viral = con.execute("SELECT count(*) FROM videos WHERE is_viral = 1").fetchone()[0]
    con.close()

    print(f"✅ DuckDB Viral Videos criado com sucesso em: {DB_PATH}")
    print(f"📊 Total de Vídeos Analisados: {total_vids:,}")
    print(f"👀 Total de Visualizações Mapeadas: {total_views:,}")
    print(f"🔥 Vídeos com Distribuição Viral: {total_viral:,} ({total_viral/total_vids*100:.1f}%)")
    return df

if __name__ == "__main__":
    generate_video_dataset()
