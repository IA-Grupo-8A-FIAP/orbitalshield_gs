# backtesting/backtest_may2024.py
"""
OrbitalShield — Backtesting: Evento Extremo Maio/2024
======================================================
ATENÇÃO: Este script usa o TEST SET (mai/2024) — usar UMA ÚNICA VEZ.
         Não ajustar nenhum parâmetro após ver os resultados.

O evento de maio/2024 foi a maior tempestade geomagnética em 20 anos:
  - Kp máximo: 9.0 (escala 0-9)
  - Dst mínimo: ~-412 nT
  - Impacto real: falhas em sistemas GNSS/RTK em todo o mundo

Este script responde: o modelo teria alertado ANTES do pico?

Fluxo:
  1. Carrega mai/2024 do banco (source=omniweb)
  2. Aplica predict_batch — mesmas features do treino
  3. Compara OGII previsto vs evolução real do Kp/Dst
  4. Gera gráfico de linha temporal com alertas
  5. Calcula métricas de antecipação

Resultado salvo em: backtesting/results/
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
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import SessionLocal
from db.models import SpaceWeatherRaw
from features.ipo import IPOThresholds
from model.predict import load_artifacts, predict_batch, ogii_to_alert

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "backtesting" / "results"
THRESH_FILE = ROOT / "sprint0" / "thresholds.json"


# ─── Carga do test set ────────────────────────────────────────────────────────

def load_test_set() -> pd.DataFrame:
    """
    Carrega maio/2024 completo do banco.
    Inclui abril/2024 como contexto para rolling features —
    apenas maio é avaliado, abril serve só de warmup.
    """
    session = SessionLocal()
    try:
        rows = (
            session.query(SpaceWeatherRaw)
            .filter(
                SpaceWeatherRaw.source == "omniweb",
                SpaceWeatherRaw.collected_at >= "2024-04-25",  # warmup
                SpaceWeatherRaw.collected_at <= "2024-05-31",
            )
            .order_by(SpaceWeatherRaw.collected_at)
            .all()
        )
    finally:
        session.close()

    df = pd.DataFrame([{
        "collected_at":      r.collected_at,
        "kp":                r.kp,
        "bz_nT":             r.bz_nT,
        "dst":               r.dst,
        "ae_index":          r.ae_index,
        "solar_wind_speed":  r.solar_wind_speed,
    } for r in rows])

    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    logger.info(f"Test set carregado: {len(df)} registros (25/abr → 31/mai/2024)")
    return df


# ─── Backtesting principal ────────────────────────────────────────────────────

def run_backtest():
    print("\n" + "=" * 56)
    print("  OrbitalShield — Backtesting Maio/2024")
    print("  ⚠️  TEST SET — usar uma única vez")
    print("=" * 56)

    # Carrega modelo e thresholds
    model, thresholds = load_artifacts()

    # Carrega dados
    raw_df = load_test_set()
    if raw_df.empty:
        print("ERRO: sem dados de mai/2024 no banco.")
        print("Execute: python -c \"from ingestion.omniweb_loader import load_historical; load_historical(2024, 2024)\"")
        sys.exit(1)

    # Predição em batch (inclui warmup de abril)
    result_df = predict_batch(raw_df, model, thresholds, save_to_db=False)

    # Filtra apenas maio/2024 para avaliação
    may_df = result_df[
        result_df["collected_at"] >= "2024-05-01"
    ].copy().reset_index(drop=True)

    print(f"\nRegistros avaliados (mai/2024): {len(may_df)}")

    # ── Métricas de classificação ─────────────────────────────────────────────
    y_true = may_df["ipo_future"].dropna().astype(int)
    y_pred = may_df.loc[y_true.index, "ipo_class_pred"].astype(int)

    f1_macro  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report    = classification_report(y_true, y_pred, zero_division=0)
    recall_c3 = classification_report(y_true, y_pred, output_dict=True,
                                      zero_division=0).get("3", {}).get("recall", 0)

    print(f"\n{'─'*50}")
    print("  Métricas no Test Set (mai/2024)")
    print(f"{'─'*50}")
    print(f"  F1-macro:         {f1_macro:.4f}")
    print(f"  Recall classe 3:  {recall_c3:.4f}")
    print(f"\n{report}")

    # ── Análise de antecipação ────────────────────────────────────────────────
    print(f"{'─'*50}")
    print("  Análise de Antecipação do Pico")
    print(f"{'─'*50}")

    # Pico real: quando Kp atingiu máximo
    idx_kp_max  = may_df["kp"].idxmax()
    dt_kp_max   = may_df.loc[idx_kp_max, "collected_at"]
    kp_max_val  = may_df.loc[idx_kp_max, "kp"]

    # Mínimo de Dst (pode não coincidir com pico de Kp)
    idx_dst_min = may_df["dst"].idxmin()
    dt_dst_min  = may_df.loc[idx_dst_min, "collected_at"]
    dst_min_val = may_df.loc[idx_dst_min, "dst"]

    # Primeiro alerta CRÍTICO (alert_level == "CRÍTICO") antes do pico
    # Usa alert_level para alinhar com a lógica de ogii_to_alert em predict.py
    criticos_antes = may_df[
        (may_df["collected_at"] <= dt_kp_max)
        & (may_df["alert_level"] == "CRÍTICO")
    ]

    primeiro_alerta  = None
    antecipacao_h    = None
    if not criticos_antes.empty:
        primeiro_alerta = criticos_antes.iloc[0]["collected_at"]
        antecipacao_h   = (dt_kp_max - primeiro_alerta).total_seconds() / 3600
        print(f"  Pico real (Kp={kp_max_val:.1f}): {dt_kp_max.strftime('%d/%m %H:%M')} UTC")
        print(f"  Primeiro alerta CRÍTICO:  {primeiro_alerta.strftime('%d/%m %H:%M')} UTC")
        print(f"  Antecipação:              {antecipacao_h:.1f} horas antes do pico ✅")
    else:
        print(f"  Pico real (Kp={kp_max_val:.1f}): {dt_kp_max.strftime('%d/%m %H:%M')} UTC")
        print("  Nenhum alerta CRÍTICO antes do pico ❌")

    # Distribuição dos alertas em maio
    alert_dist = may_df["alert_level"].value_counts()
    print(f"\n  Distribuição de alertas em mai/2024:")
    for level, count in alert_dist.items():
        pct = count / len(may_df) * 100
        icon = ogii_to_alert(
            {"BAIXO": 10, "MODERADO": 35, "ALTO": 60, "CRÍTICO": 85}[level]
        )["icon"]
        print(f"    {icon} {level:10s}: {count:4d} horas ({pct:.1f}%)")

    # ── Gráficos ──────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_timeline(
        may_df, dt_kp_max, dt_dst_min, dst_min_val,
        primeiro_alerta, antecipacao_h,
        f1_macro, recall_c3, kp_max_val
    )
    _plot_confusion_detail(y_true, y_pred)

    # ── Salva resultados ──────────────────────────────────────────────────────
    summary = {
        "test_period":     "2024-05-01 → 2024-05-31",
        "n_records":       len(may_df),
        "f1_macro":        round(f1_macro, 4),
        "recall_critical": round(recall_c3, 4),
        "kp_max":          float(kp_max_val),
        "dt_kp_max":       str(dt_kp_max),
        "dt_dst_min":      str(dt_dst_min),
        "dst_min":         float(dst_min_val),
        "antecipacao_h":   round(antecipacao_h, 1) if antecipacao_h is not None else None,
        "primeiro_alerta": str(primeiro_alerta) if primeiro_alerta is not None else None,
        "alert_distribution": alert_dist.to_dict(),
    }
    out_json = RESULTS_DIR / "backtest_may2024_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Resumo salvo: {out_json}")

    print(f"\n  Resultados salvos em: {RESULTS_DIR}/")
    print("=" * 56)

    return summary


# ─── Gráficos ─────────────────────────────────────────────────────────────────

def _plot_timeline(
    df: pd.DataFrame,
    dt_peak: pd.Timestamp,
    dt_dst_min: pd.Timestamp,
    dst_min_val: float,
    primeiro_alerta: pd.Timestamp | None,
    antecipacao_h: float | None,
    f1_macro: float,
    recall_c3: float,
    kp_max: float,
):
    """
    Gráfico principal: OGII + Kp + Dst ao longo de maio/2024.

    Melhorias v2:
      1. Linha do primeiro alerta CRÍTICO + anotação de antecipação em horas
      2. Sombreado da janela de antecipação [primeiro_crítico → pico_kp]
      3. OGII bruto (fino) + OGII suavizado 24h (espesso) — métricas usam bruto
      4. Título dinâmico com F1-macro, recall crítico e antecipação
      5. Marcação separada do mínimo de Dst (não coincide com pico Kp)
      6. Legenda de Kp=7 no painel 2
      7. Nota de rodapé: TEST SET
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    # Título dinâmico com métricas do backtest
    antecipacao_str = f"{antecipacao_h:.0f}h" if antecipacao_h is not None else "N/A"
    fig.suptitle(
        f"OrbitalShield — Backtesting Evento Extremo Maio/2024\n"
        f"F1-macro: {f1_macro:.4f}  |  Recall crítico: {recall_c3:.4f}  |  "
        f"Antecipação: {antecipacao_str}  |  Kp max: {kp_max:.0f}  |  Dst min: {dst_min_val:.0f} nT",
        fontsize=12, fontweight="bold", y=0.995
    )

    ts = df["collected_at"]

    # OGII suavizado 24h (apenas visual — métricas usam bruto)
    ogii_smooth = df["ogii"].rolling(window=24, center=False, min_periods=1).mean()

    # ── Painel 1: OGII ────────────────────────────────────────────────────────
    ax1 = axes[0]

    # Faixas de alerta
    ax1.fill_between(ts, 0,   25,  alpha=0.07, color="#27ae60")
    ax1.fill_between(ts, 25,  50,  alpha=0.07, color="#f39c12")
    ax1.fill_between(ts, 50,  75,  alpha=0.07, color="#e67e22")
    ax1.fill_between(ts, 75,  100, alpha=0.07, color="#c0392b")

    # OGII bruto — fino e transparente
    ax1.plot(ts, df["ogii"], color="#2c3e50", linewidth=0.8,
             alpha=0.35, label="OGII bruto (base das métricas)")

    # OGII suavizado — espesso e legível
    ax1.plot(ts, ogii_smooth, color="#2c3e50", linewidth=2.2,
             alpha=0.9, label="OGII suavizado 24h (visual)")

    # Limiar CRÍTICO
    ax1.axhline(75, color="#c0392b", linestyle="--", linewidth=0.9, alpha=0.7,
                label="limiar ALTO/CRÍTICO (75)")

    # Linha do pico de Kp
    ax1.axvline(dt_peak, color="#c0392b", linestyle="--", linewidth=1.4,
                label=f"Pico Kp={kp_max:.0f} ({dt_peak.strftime('%d/%m')})")

    # Linha do primeiro alerta CRÍTICO + sombreado de antecipação
    if primeiro_alerta is not None:
        ax1.axvline(primeiro_alerta, color="#e67e22", linestyle="-",
                    linewidth=1.6, label=f"1º alerta CRÍTICO ({primeiro_alerta.strftime('%d/%m %H:%M')})")
        ax1.axvspan(primeiro_alerta, dt_peak, alpha=0.07, color="#e67e22",
                    label=f"Janela antecipação ({antecipacao_h:.0f}h)")
        # Anotação da antecipação no meio da janela
        mid_dt = primeiro_alerta + (dt_peak - primeiro_alerta) / 2
        ax1.annotate(
            f"+{antecipacao_h:.0f}h",
            xy=(mid_dt, 95),
            ha="center", va="top",
            fontsize=9, fontweight="bold", color="#e67e22",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#e67e22", alpha=0.85)
        )

    ax1.set_ylabel("OGII (0–100)", fontsize=10)
    ax1.set_ylim(0, 108)
    ax1.legend(loc="lower left", fontsize=8, ncol=2,
               framealpha=0.85, edgecolor="#cccccc")
    ax1.set_title("Índice Operacional GNSS (OGII)", fontsize=10, pad=4)

    # Labels das faixas à direita (fora do eixo de dados)
    ax1_r = ax1.twinx()
    ax1_r.set_ylim(0, 108)
    ax1_r.set_yticks([12, 37, 62, 87])
    ax1_r.set_yticklabels(
        ["BAIXO", "MODERADO", "ALTO", "CRÍTICO"],
        fontsize=7
    )
    ax1_r.tick_params(axis="y", length=0)
    for tick, color in zip(ax1_r.get_yticklabels(),
                           ["#27ae60", "#f39c12", "#e67e22", "#c0392b"]):
        tick.set_color(color)

    # ── Painel 2: Kp ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    colors_kp = [
        "#27ae60" if k < 3 else
        "#f39c12" if k < 5 else
        "#e67e22" if k < 7 else
        "#c0392b"
        for k in df["kp"]
    ]
    ax2.bar(ts, df["kp"], width=0.04, color=colors_kp, alpha=0.85)
    ax2.axvline(dt_peak, color="#c0392b", linestyle="--", linewidth=1.4)
    if primeiro_alerta is not None:
        ax2.axvline(primeiro_alerta, color="#e67e22", linestyle="-", linewidth=1.4)

    # Linha Kp=7 COM legenda
    ax2.axhline(7, color="#c0392b", linestyle=":", linewidth=1.0, alpha=0.7,
                label="Kp = 7 (limiar tempestade forte)")
    ax2.set_ylabel("Kp", fontsize=10)
    ax2.set_ylim(0, 10.5)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.85,
               edgecolor="#cccccc")
    ax2.set_title("Índice Kp (Atividade Geomagnética)", fontsize=10, pad=4)

    # ── Painel 3: Dst ─────────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(ts, df["dst"], color="#8e44ad", linewidth=1.4, label="Dst")
    ax3.fill_between(ts, df["dst"], 0, where=(df["dst"] < 0),
                     alpha=0.15, color="#8e44ad")

    # Linha do pico de Kp
    ax3.axvline(dt_peak, color="#c0392b", linestyle="--", linewidth=1.4,
                label=f"Pico Kp={kp_max:.0f} ({dt_peak.strftime('%d/%m')})")

    # Linha do mínimo de Dst — separada do pico de Kp
    ax3.axvline(dt_dst_min, color="#8e44ad", linestyle="-.", linewidth=1.4,
                label=f"Dst mín. = {dst_min_val:.0f} nT ({dt_dst_min.strftime('%d/%m %H:%M')})")

    if primeiro_alerta is not None:
        ax3.axvline(primeiro_alerta, color="#e67e22", linestyle="-", linewidth=1.4)

    ax3.axhline(-100, color="#c0392b", linestyle=":", linewidth=0.9,
                alpha=0.6, label="Limiar tempestade intensa (−100 nT)")
    ax3.set_ylabel("Dst (nT)", fontsize=10)
    ax3.set_title("Índice Dst (Resposta Magnetosférica)", fontsize=10, pad=4)
    ax3.legend(loc="lower left", fontsize=8, ncol=2,
               framealpha=0.85, edgecolor="#cccccc")

    # Formatação eixo X
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax3.set_xlabel("Data (mai/2024)", fontsize=10)

    # Nota de rodapé — TEST SET
    fig.text(
        0.5, 0.002,
        "⚠️  TEST SET — não usar para ajuste de parâmetros",
        ha="center", va="bottom", fontsize=8,
        color="#888888", style="italic"
    )

    plt.tight_layout(rect=[0, 0.012, 1, 1])

    # Salva PNG e PDF
    out_png = RESULTS_DIR / "backtest_timeline.png"
    out_pdf = RESULTS_DIR / "backtest_timeline.pdf"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    logger.info(f"Timeline salvo: {out_png} | {out_pdf}")


def _plot_confusion_detail(y_true, y_pred):
    """Matriz de confusão no test set."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    lbls   = ["Baixo", "Moderado", "Alto", "Crítico"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_pct, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=lbls, yticklabels=lbls, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(
        "Matriz de Confusão — Test Set (mai/2024)\n"
        "Evento Extremo Kp=9",
        pad=10
    )
    fig.text(
        0.5, 0.01,
        "⚠️  TEST SET — não usar para ajuste de parâmetros",
        ha="center", va="bottom", fontsize=8,
        color="#888888", style="italic"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = RESULTS_DIR / "backtest_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix salva: {out}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_backtest()