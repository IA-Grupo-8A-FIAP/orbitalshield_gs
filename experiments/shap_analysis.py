# experiments/shap_analysis.py
"""
OrbitalShield — SHAP Values: Explicabilidade do Modelo
=======================================================
Gera análise SHAP do XGBoost treinado para explicar:
  - Quais features mais impactam cada classe de risco
  - Como cada feature influencia individualmente a predição
  - Summary plot e dependence plots das top features

Regras:
  - Usa o modelo já treinado em model/artifacts/
  - Avalia no val set (jan-abr/2024) — test set preservado
  - NÃO retreina o modelo

Uso:
    pip install shap
    python experiments/shap_analysis.py

Saídas em experiments/:
  - shap_summary.png          (beeswarm — visão geral)
  - shap_bar.png              (importância média |SHAP|)
  - shap_class_critical.png   (SHAP específico classe CRÍTICO)
  - shap_results.json         (top features por classe)
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import SessionLocal
from db.models import SpaceWeatherRaw
from features.engineering import build_features, FEATURE_COLS
from features.ipo import IPOThresholds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR  = ROOT / "experiments"
ARTIFACT_DIR = ROOT / "model" / "artifacts"
THRESH_FILE  = ROOT / "sprint0" / "thresholds.json"

CLASS_NAMES = ["Baixo", "Moderado", "Alto", "Crítico"]


# ─── Verifica dependência SHAP ────────────────────────────────────────────────

try:
    import shap
except ImportError:
    print("ERRO: shap não instalado.")
    print("Execute: pip install shap")
    sys.exit(1)


# ─── Carga ────────────────────────────────────────────────────────────────────

def load_val_set() -> pd.DataFrame:
    session = SessionLocal()
    try:
        rows = (
            session.query(SpaceWeatherRaw)
            .filter(
                SpaceWeatherRaw.source == "omniweb",
                SpaceWeatherRaw.collected_at >= "2024-01-01",
                SpaceWeatherRaw.collected_at <= "2024-04-30",
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


# ─── Nomes legíveis das features ──────────────────────────────────────────────

FEATURE_LABELS = {
    "kp":                 "Kp (atividade geomagnética)",
    "bz_nT":              "Bz (campo magnético sul)",
    "bz_min_3h":          "Bz mín. 3h anteriores",
    "dst":                "Dst (resposta magnetosférica)",
    "ae_index":           "AE (atividade auroral)",
    "southward_duration": "Duração campo sul (h)",
    "hour_sin":           "Hora — seno (ciclo diurno)",
    "hour_cos":           "Hora — cosseno (ciclo diurno)",
    "solar_wind_speed":   "Vento solar (km/s)",
    "kp_lag_3h":          "Kp lag 3h",
    "kp_mean_6h":         "Kp média 6h",
    "kp_x_bz_neg":        "Kp × |Bz negativo|",
    "amas_factor":        "AMAS (fator regional)",
}

LABELS_SHORT = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_shap_bar(shap_values, X_df: pd.DataFrame, out_path: Path):
    """Importância média |SHAP| por feature (todas as classes)."""
    # Média do valor absoluto SHAP por feature, somando sobre classes
    mean_abs = np.abs(shap_values).mean(axis=0)  # (n_samples, n_features)
    # Se shap_values for 3D (n_samples, n_features, n_classes), média sobre classes
    if mean_abs.ndim == 2:
        mean_abs = mean_abs.mean(axis=1)

    df_imp = pd.DataFrame({
        "feature": LABELS_SHORT,
        "importance": mean_abs,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f6f8fa")
    ax.set_facecolor("#f6f8fa")

    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(df_imp)))
    bars = ax.barh(df_imp["feature"], df_imp["importance"],
                   color=colors, alpha=0.85)

    for bar, v in zip(bars, df_imp["importance"]):
        ax.text(bar.get_width() + 0.0002,
                bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8)

    ax.set_title("SHAP — Importância Média |SHAP| por Feature\n"
                 "Val Set (jan–abr/2024)", fontsize=12,
                 fontweight="bold", pad=10)
    ax.set_xlabel("Importância SHAP média (|valor|)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP bar salvo: {out_path}")


def plot_shap_class(shap_vals_class, X_df: pd.DataFrame,
                    class_name: str, out_path: Path):
    """Beeswarm SHAP para uma classe específica."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f6f8fa")

    # Ordena por importância média
    mean_abs = np.abs(shap_vals_class).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:10]  # top 10

    shap_plot = shap_vals_class[:, order]
    feat_plot = X_df.iloc[:, order]
    labels    = [LABELS_SHORT[i] for i in order]

    # Scatter manual por feature
    y_positions = np.arange(len(order))
    for yi, (col_shap, col_feat, lbl) in enumerate(
            zip(shap_plot.T, feat_plot.T.values, labels)):
        # Normaliza feature para cor
        feat_norm = (col_feat - col_feat.min()) / (
            col_feat.max() - col_feat.min() + 1e-9)
        sc = ax.scatter(col_shap,
                        np.full_like(col_shap, yi) + np.random.normal(0, 0.08, len(col_shap)),
                        c=feat_norm, cmap="RdBu_r", alpha=0.4, s=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("SHAP value (impacto na predição)", fontsize=10)
    ax.set_title(f"SHAP — Classe '{class_name}'\n"
                 f"Azul = valor baixo da feature  |  Vermelho = valor alto",
                 fontsize=11, fontweight="bold", pad=10)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("Valor da feature (normalizado)", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("#f6f8fa")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP {class_name} salvo: {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 56)
    print("  OrbitalShield — SHAP Values")
    print("  Modelo: model/artifacts/xgboost_model.joblib")
    print("=" * 56)

    # Carrega modelo
    model_path = ARTIFACT_DIR / "xgboost_model.joblib"
    if not model_path.exists():
        print(f"ERRO: {model_path} não encontrado.")
        print("Execute model/train.py primeiro.")
        sys.exit(1)

    model = joblib.load(model_path)
    logger.info(f"Modelo carregado: {model_path}")

    # Carrega val set
    thresh_data = json.loads(THRESH_FILE.read_text())
    thresholds  = IPOThresholds(
        p25=thresh_data["p25"],
        p50=thresh_data["p50"],
        p75=thresh_data["p75"],
    )

    print("Carregando val set (jan–abr/2024)...")
    raw_val = load_val_set()
    if raw_val.empty:
        print("ERRO: banco sem dados.")
        sys.exit(1)

    val_df = build_features(raw_val, thresholds=thresholds)
    val_df = val_df.dropna(subset=["ipo_future"] + FEATURE_COLS)
    X_val  = val_df[FEATURE_COLS]

    # Amostra para SHAP (máx 500 para velocidade)
    n_sample = min(500, len(X_val))
    X_sample = X_val.sample(n=n_sample, random_state=42)
    logger.info(f"SHAP calculado em {n_sample} amostras do val set")

    # ── Calcula SHAP ──────────────────────────────────────────────────────────
    print(f"\nCalculando SHAP values ({n_sample} amostras)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # shap_values: lista de n_classes arrays (n_samples, n_features)

    print("✅ SHAP calculado")

    # ── Gráficos ──────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Gerando gráficos...")

    # Normaliza shap_values para lista de arrays por classe — usado em todos os plots
    if isinstance(shap_values, list):
        shap_by_class = shap_values
        shap_all      = np.stack(shap_values, axis=2)
    else:
        shap_by_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        shap_all      = shap_values

    mean_abs_all = np.abs(shap_all).mean(axis=(0, 2))

    # 1. Bar plot — importância geral

    # Garante shape correto
    mean_abs_all = mean_abs_all.flatten()
    assert len(mean_abs_all) == len(FEATURE_COLS), (
        f"Shape mismatch: {len(mean_abs_all)} vs {len(FEATURE_COLS)}"
    )

    df_imp = pd.DataFrame({
        "feature":    LABELS_SHORT,
        "importance": mean_abs_all,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f6f8fa")
    ax.set_facecolor("#f6f8fa")
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(df_imp)))
    bars = ax.barh(df_imp["feature"], df_imp["importance"],
                   color=colors, alpha=0.85)
    for bar, v in zip(bars, df_imp["importance"]):
        ax.text(bar.get_width() + 0.0001,
                bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8)
    ax.set_title("SHAP — Importância Média por Feature (todas as classes)\n"
                 "Val Set (jan–abr/2024)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Importância SHAP média |valor|", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("shap_bar.png salvo")

    # 2. Beeswarm da classe CRÍTICO (classe 3 — mais importante)
    plot_shap_class(
        shap_by_class[3], X_sample,
        "Crítico", RESULTS_DIR / "shap_class_critical.png"
    )

    # 3. Beeswarm da classe BAIXO (classe 0 — contraste)
    plot_shap_class(
        shap_by_class[0], X_sample,
        "Baixo", RESULTS_DIR / "shap_class_low.png"
    )

    # ── Top features por classe ───────────────────────────────────────────────
    top_by_class = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mean_abs_cls = np.abs(shap_by_class[cls_idx]).mean(axis=0)
        order        = np.argsort(mean_abs_cls)[::-1][:5]
        top_by_class[cls_name] = [
            {"feature": FEATURE_COLS[i],
             "label":   LABELS_SHORT[i],
             "mean_abs_shap": round(float(mean_abs_cls[i]), 4)}
            for i in order
        ]

    # ── Salva JSON ────────────────────────────────────────────────────────────
    summary = {
        "val_period":    "2024-01-01 → 2024-04-30",
        "n_samples":     n_sample,
        "test_set_used": False,
        "top_features_by_class": top_by_class,
        "global_ranking": [
            {"feature": FEATURE_COLS[i],
             "label":   LABELS_SHORT[i],
             "mean_abs_shap": round(float(mean_abs_all[i]), 4)}
            for i in np.argsort(mean_abs_all)[::-1]
        ],
    }
    out_json = RESULTS_DIR / "shap_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info(f"Resultados SHAP: {out_json}")

    # ── Resumo no terminal ────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  TOP 5 FEATURES — Classe CRÍTICO")
    print("=" * 56)
    for i, f in enumerate(top_by_class["Crítico"], 1):
        print(f"  {i}. {f['label']:35s} SHAP={f['mean_abs_shap']:.4f}")

    print("\n" + "=" * 56)
    print("  RANKING GLOBAL (todas as classes)")
    print("=" * 56)
    for i, f in enumerate(summary["global_ranking"][:8], 1):
        print(f"  {i}. {f['label']:35s} SHAP={f['mean_abs_shap']:.4f}")

    print(f"\n  Gráficos salvos em: {RESULTS_DIR}/")
    print("=" * 56)
    return summary


if __name__ == "__main__":
    run()