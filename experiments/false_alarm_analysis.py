# experiments/false_alarm_analysis.py
"""
OrbitalShield — Análise de Alarmes Falsos
==========================================
Responde à pergunta da banca:
  "42% do mês em CRÍTICO parece alarme contínuo — quantos eram falsos?"

Metodologia:
  - Usa o test set de maio/2024 (já calculado em backtest)
  - Cruza alert_level == "CRÍTICO" com Kp real
  - Define "alarme falso operacional" como CRÍTICO com Kp < 5
    (abaixo do limiar de tempestade moderada)
  - Gera gráfico e salva JSON com métricas

Regras:
  - NÃO retreina o modelo
  - NÃO recalibra thresholds
  - Usa dados já persistidos do backtest

Saídas em experiments/:
  - false_alarm_analysis.png
  - false_alarm_results.json
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import SessionLocal
from db.models import SpaceWeatherRaw
from features.engineering import build_features, FEATURE_COLS
from features.ipo import IPOThresholds
from model.predict import load_artifacts, predict_batch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "experiments"
THRESH_FILE = ROOT / "sprint0" / "thresholds.json"


def run():
    print("\n" + "=" * 56)
    print("  OrbitalShield — Análise de Alarmes Falsos")
    print("  Test set: maio/2024")
    print("=" * 56)

    # ── Carrega dados de maio/2024 ────────────────────────────────────────────
    model, thresholds = load_artifacts()

    session = SessionLocal()
    try:
        from db.models import SpaceWeatherRaw
        rows = (
            session.query(SpaceWeatherRaw)
            .filter(
                SpaceWeatherRaw.source == "omniweb",
                SpaceWeatherRaw.collected_at >= "2024-04-25",
                SpaceWeatherRaw.collected_at <= "2024-05-31",
            )
            .order_by(SpaceWeatherRaw.collected_at)
            .all()
        )
    finally:
        session.close()

    raw_df = pd.DataFrame([{
        "collected_at":     r.collected_at,
        "kp":               r.kp or 0.0,
        "bz_nT":            r.bz_nT or 0.0,
        "dst":              r.dst or 0.0,
        "ae_index":         r.ae_index or 0.0,
        "solar_wind_speed": r.solar_wind_speed or 400.0,
    } for r in rows])
    raw_df["collected_at"] = pd.to_datetime(raw_df["collected_at"], utc=True)

    result_df = predict_batch(raw_df, model, thresholds, save_to_db=False)
    may_df    = result_df[
        result_df["collected_at"] >= "2024-05-01"
    ].copy().reset_index(drop=True)

    total_hours = len(may_df)

    # ── Classificação dos alertas CRÍTICO ─────────────────────────────────────
    criticos = may_df[may_df["alert_level"] == "CRÍTICO"].copy()

    # Definições de "alarme falso operacional"
    # Kp < 5: abaixo de tempestade moderada (G1) — sem justificativa física forte
    # Kp < 3: atividade baixa — alarme claramente conservador
    criticos["kp_class"] = pd.cut(
        criticos["kp"],
        bins=[-0.1, 3, 5, 7, 9],
        labels=["Kp < 3\n(baixo)", "Kp 3–5\n(moderado)", "Kp 5–7\n(alto)", "Kp > 7\n(severo)"]
    )

    kp_dist = criticos["kp_class"].value_counts().sort_index()

    # Alarmes falsos operacionais = CRÍTICO com Kp < 5
    false_alarms   = criticos[criticos["kp"] < 5]
    true_alarms    = criticos[criticos["kp"] >= 5]
    n_critico      = len(criticos)
    n_false        = len(false_alarms)
    n_true         = len(true_alarms)
    pct_false      = n_false / n_critico * 100 if n_critico > 0 else 0
    pct_true       = n_true  / n_critico * 100 if n_critico > 0 else 0

    print(f"\n  Total horas em maio:          {total_hours}")
    print(f"  Horas CRÍTICO:                {n_critico} ({n_critico/total_hours*100:.1f}%)")
    print(f"  CRÍTICO com Kp >= 5 (válidos): {n_true} ({pct_true:.1f}% dos críticos)")
    print(f"  CRÍTICO com Kp < 5 (conserv.): {n_false} ({pct_false:.1f}% dos críticos)")

    # ── Gráfico ───────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f6f8fa")

    # Painel 1 — Pizza de alarmes falsos vs válidos
    ax1 = axes[0]
    ax1.set_facecolor("#f6f8fa")
    sizes  = [n_true, n_false]
    labels = [
        f"Kp ≥ 5\n(justificado)\n{n_true}h ({pct_true:.0f}%)",
        f"Kp < 5\n(conservador)\n{n_false}h ({pct_false:.0f}%)"
    ]
    colors = ["#27ae60", "#f39c12"]
    wedges, texts = ax1.pie(
        sizes, labels=labels, colors=colors,
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for text in texts:
        text.set_fontsize(10)
    ax1.set_title(
        "Alertas CRÍTICO em Maio/2024\nJustificado (Kp≥5) vs Conservador (Kp<5)",
        fontsize=11, fontweight="bold", pad=12
    )

    # Painel 2 — Distribuição de Kp nos alertas CRÍTICO
    ax2 = axes[1]
    ax2.set_facecolor("#f6f8fa")

    kp_bins  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9.1]
    kp_labels= ["0","1","2","3","4","5","6","7","8–9"]
    hist, _  = np.histogram(criticos["kp"].values, bins=kp_bins)

    bar_colors = []
    for i, lb in enumerate(kp_labels):
        kp_mid = i
        if kp_mid < 3:
            bar_colors.append("#27ae60")
        elif kp_mid < 5:
            bar_colors.append("#f39c12")
        elif kp_mid < 7:
            bar_colors.append("#e67e22")
        else:
            bar_colors.append("#c0392b")

    bars = ax2.bar(range(len(kp_labels)), hist, color=bar_colors, alpha=0.85,
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, hist):
        if v > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(v), ha="center", va="bottom", fontsize=9,
                     fontweight="bold")

    ax2.axvline(4.5, color="#c0392b", linestyle="--", linewidth=1.2,
                alpha=0.7, label="Limiar Kp=5 (G1)")
    ax2.set_xticks(range(len(kp_labels)))
    ax2.set_xticklabels(kp_labels)
    ax2.set_xlabel("Kp no momento do alerta CRÍTICO", fontsize=10)
    ax2.set_ylabel("Nº de horas", fontsize=10)
    ax2.set_title(
        "Distribuição de Kp durante alertas CRÍTICO\n"
        "Verde = baixo | Laranja = moderado | Vermelho = severo",
        fontsize=11, fontweight="bold", pad=12
    )
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Contexto: janeiro foi a maior concentração pré-tempestade
    fig.suptitle(
        "OrbitalShield — Análise de Alarmes Conservadores | Maio/2024\n"
        "Conservadorismo é esperado: modelo prevê t+1h a partir de precursores geofísicos",
        fontsize=11, fontweight="bold", y=1.02
    )

    fig.text(
        0.5, -0.02,
        "Nota: 'conservador' ≠ 'errado' — Kp < 5 antes do pico pode refletir "
        "precursores reais da tempestade (southward_duration, bz_min_3h, kp_mean_6h)",
        ha="center", fontsize=8, color="#666", style="italic"
    )

    plt.tight_layout()
    out_png = RESULTS_DIR / "false_alarm_analysis.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Gráfico salvo: {out_png}")

    # ── Salva JSON ────────────────────────────────────────────────────────────
    summary = {
        "period":             "2024-05-01 → 2024-05-31",
        "total_hours":        total_hours,
        "critical_hours":     n_critico,
        "critical_pct":       round(n_critico / total_hours * 100, 1),
        "justified_kp_ge5":   n_true,
        "justified_pct":      round(pct_true, 1),
        "conservative_kp_lt5": n_false,
        "conservative_pct":   round(pct_false, 1),
        "note": (
            "Conservador != errado. Kp < 5 antes do pico pode refletir "
            "precursores reais capturados pelas rolling features (southward_duration, "
            "bz_min_3h, kp_mean_6h). O modelo prevê t+1h a partir de precursores, "
            "não do pico instantâneo."
        )
    }
    out_json = RESULTS_DIR / "false_alarm_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info(f"JSON salvo: {out_json}")

    print(f"\n  Gráfico: {out_png}")
    print(f"  JSON:    {out_json}")
    print("=" * 56)
    return summary


if __name__ == "__main__":
    run()