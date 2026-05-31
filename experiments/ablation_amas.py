# experiments/ablation_amas.py
"""
OrbitalShield — Ablation Study: AMAS Factor
=============================================
Responde: o fator AMAS (amas_factor) contribui ou prejudica o modelo?

Metodologia:
  1. Treina modelo COM amas_factor (configuração atual)
  2. Treina modelo SEM amas_factor (feature removida)
  3. Compara F1-macro e recall crítico no val set (jan-abr/2024)
  4. Gera gráfico comparativo e salva resultado em experiments/

Regras:
  - Usa os mesmos thresholds congelados do Sprint 0
  - NÃO toca no test set (mai/2024)
  - NÃO altera artefatos do modelo principal
  - Resultado determina se AMAS deve ser mantido ou removido

Uso:
    python experiments/ablation_amas.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import SessionLocal
from db.models import SpaceWeatherRaw
from features.engineering import build_features, FEATURE_COLS
from features.ipo import IPOThresholds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "experiments"
THRESH_FILE = ROOT / "sprint0" / "thresholds.json"

# Parâmetros idênticos ao train.py para comparação justa
XGBOOST_PARAMS = {
    "n_estimators":      400,
    "max_depth":         5,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "objective":         "multi:softprob",
    "num_class":         4,
    "eval_metric":       "mlogloss",
    "random_state":      42,
    "use_label_encoder": False,
    "verbosity":         0,
}
EARLY_STOPPING = 25


# ─── Carga de dados ───────────────────────────────────────────────────────────

def load_period(start: str, end: str) -> pd.DataFrame:
    session = SessionLocal()
    try:
        rows = (
            session.query(SpaceWeatherRaw)
            .filter(
                SpaceWeatherRaw.source == "omniweb",
                SpaceWeatherRaw.collected_at >= start,
                SpaceWeatherRaw.collected_at <= end,
            )
            .order_by(SpaceWeatherRaw.collected_at)
            .all()
        )
    finally:
        session.close()
    return pd.DataFrame([{
        "collected_at":     r.collected_at,
        "kp":               r.kp,
        "bz_nT":            r.bz_nT,
        "dst":              r.dst,
        "ae_index":         r.ae_index,
        "solar_wind_speed": r.solar_wind_speed,
    } for r in rows])


# ─── Treino e avaliação ───────────────────────────────────────────────────────

def train_and_eval(
    X_train, y_train,
    X_val,   y_val,
    feature_cols: list,
    label: str,
) -> dict:
    """Treina um XGBoost e retorna métricas."""
    model = xgb.XGBClassifier(
        **XGBOOST_PARAMS,
        early_stopping_rounds=EARLY_STOPPING
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_pred    = model.predict(X_val)
    f1_macro  = f1_score(y_val, y_pred, average="macro", zero_division=0)
    report    = classification_report(y_val, y_pred,
                                      output_dict=True, zero_division=0)
    recall_c3 = report.get("3", {}).get("recall", 0.0)

    print(f"\n  [{label}]")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  F1-macro:         {f1_macro:.4f}")
    print(f"  Recall crítico:   {recall_c3:.4f}")
    print(f"  Melhor iteração:  {model.best_iteration}")

    return {
        "label":           label,
        "n_features":      len(feature_cols),
        "feature_cols":    feature_cols,
        "f1_macro":        round(f1_macro, 4),
        "recall_critical": round(recall_c3, 4),
        "best_iteration":  int(model.best_iteration),
    }


# ─── Gráfico comparativo ──────────────────────────────────────────────────────

def plot_comparison(results: list, out_path: Path):
    """Gráfico de barras comparando COM vs SEM amas_factor."""
    labels    = [r["label"] for r in results]
    f1_vals   = [r["f1_macro"] for r in results]
    rec_vals  = [r["recall_critical"] for r in results]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#f6f8fa")
    ax.set_facecolor("#f6f8fa")

    bars1 = ax.bar(x - width/2, f1_vals,  width, label="F1-macro",
                   color="#1f6feb", alpha=0.85)
    bars2 = ax.bar(x + width/2, rec_vals, width, label="Recall Crítico",
                   color="#27ae60", alpha=0.85)

    # Valores nas barras
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Métrica", fontsize=10)
    ax.set_title(
        "Ablation Study — AMAS Factor\n"
        "Comparação Val Set (jan–abr/2024)",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.legend(fontsize=9)
    ax.axhline(0.8185, color="#c0392b", linestyle="--",
               linewidth=0.8, alpha=0.6, label="F1 modelo atual (0.8185)")
    ax.text(1.02, 0.8185, "modelo atual", va="center",
            fontsize=8, color="#c0392b", transform=ax.get_yaxis_transform())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(0.5, 0.01,
             "Ablation Study — NÃO usa test set mai/2024",
             ha="center", fontsize=8, color="#888888", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Gráfico salvo: {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 56)
    print("  OrbitalShield — Ablation Study: AMAS Factor")
    print("  Conjunto: Val (jan–abr/2024) — test set preservado")
    print("=" * 56)

    # Thresholds congelados
    thresh_data = json.loads(THRESH_FILE.read_text())
    thresholds  = IPOThresholds(
        p25=thresh_data["p25"],
        p50=thresh_data["p50"],
        p75=thresh_data["p75"],
    )

    # Carrega dados
    print("\nCarregando dados...")
    raw_train = load_period("2018-01-01", "2023-12-31")
    raw_val   = load_period("2024-01-01", "2024-04-30")

    if raw_train.empty or raw_val.empty:
        print("ERRO: banco sem dados. Execute omniweb_loader.py primeiro.")
        sys.exit(1)

    # Feature engineering
    train_df = build_features(raw_train, thresholds=thresholds)
    val_df   = build_features(raw_val,   thresholds=thresholds)
    train_df = train_df.dropna(subset=["ipo_future"] + FEATURE_COLS)
    val_df   = val_df.dropna(subset=["ipo_future"]   + FEATURE_COLS)

    y_train = train_df["ipo_future"].values.astype(int)
    y_val   = val_df["ipo_future"].values.astype(int)

    # ── Experimento 1: COM amas_factor (configuração atual) ───────────────────
    print("\n" + "─" * 56)
    print("  Experimento 1 — COM amas_factor (configuração atual)")
    print("─" * 56)

    cols_with = FEATURE_COLS  # inclui amas_factor
    X_train_w = train_df[cols_with].values
    X_val_w   = val_df[cols_with].values

    res_with = train_and_eval(
        X_train_w, y_train,
        X_val_w,   y_val,
        cols_with, "COM amas_factor"
    )

    # ── Experimento 2: SEM amas_factor ────────────────────────────────────────
    print("\n" + "─" * 56)
    print("  Experimento 2 — SEM amas_factor")
    print("─" * 56)

    cols_without = [c for c in FEATURE_COLS if c != "amas_factor"]
    X_train_wo   = train_df[cols_without].values
    X_val_wo     = val_df[cols_without].values

    res_without = train_and_eval(
        X_train_wo, y_train,
        X_val_wo,   y_val,
        cols_without, "SEM amas_factor"
    )

    # ── Análise do resultado ──────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  RESULTADO DO ABLATION STUDY")
    print("=" * 56)

    delta_f1  = res_with["f1_macro"]        - res_without["f1_macro"]
    delta_rec = res_with["recall_critical"] - res_without["recall_critical"]

    print(f"\n  F1-macro:       COM={res_with['f1_macro']:.4f}  "
          f"SEM={res_without['f1_macro']:.4f}  "
          f"Δ={delta_f1:+.4f}")
    print(f"  Recall crítico: COM={res_with['recall_critical']:.4f}  "
          f"SEM={res_without['recall_critical']:.4f}  "
          f"Δ={delta_rec:+.4f}")

    THRESHOLD = 0.005  # diferença mínima considerada relevante

    if abs(delta_f1) < THRESHOLD and abs(delta_rec) < THRESHOLD:
        verdict = "NEUTRO"
        recommendation = (
            "amas_factor não apresenta contribuição relevante (|Δ| < 0.005). "
            "Recomendado: manter como hipótese experimental documentada, "
            "sem impacto no modelo principal."
        )
    elif delta_f1 > THRESHOLD or delta_rec > THRESHOLD:
        verdict = "POSITIVO"
        recommendation = (
            f"amas_factor melhora o modelo (ΔF1={delta_f1:+.4f}, "
            f"ΔRecall={delta_rec:+.4f}). "
            "Recomendado: manter no pipeline."
        )
    else:
        verdict = "NEGATIVO"
        recommendation = (
            f"amas_factor prejudica o modelo (ΔF1={delta_f1:+.4f}, "
            f"ΔRecall={delta_rec:+.4f}). "
            "Recomendado: remover do pipeline e documentar como hipótese refutada."
        )

    print(f"\n  Veredito: {verdict}")
    print(f"  → {recommendation}")

    # ── Salva resultados ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "ablation_target":  "amas_factor",
        "val_period":       "2024-01-01 → 2024-04-30",
        "test_set_used":    False,
        "with_amas":        res_with,
        "without_amas":     res_without,
        "delta_f1_macro":   round(delta_f1, 4),
        "delta_recall":     round(delta_rec, 4),
        "verdict":          verdict,
        "recommendation":   recommendation,
    }
    out_json = RESULTS_DIR / "ablation_amas_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info(f"Resultados salvos: {out_json}")

    plot_comparison(
        [res_with, res_without],
        RESULTS_DIR / "ablation_amas_comparison.png"
    )

    print(f"\n  Resultados salvos em: {RESULTS_DIR}/")
    print("=" * 56)
    return summary


if __name__ == "__main__":
    run()