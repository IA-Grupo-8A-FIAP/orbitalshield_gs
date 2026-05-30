# model/train.py
"""
OrbitalShield — Treinamento do Modelo XGBoost
===============================================
PRÉ-REQUISITO: sprint0/01_ipo_distribution.py deve ter passado.
               sprint0/thresholds.json deve existir.

Fluxo:
  1. Carrega dados do banco (2018-2023 train, jan-abr/2024 val)
  2. Aplica build_features com thresholds congelados do Sprint 0
  3. Avalia baseline heurístico no val set
  4. Treina XGBoost com early stopping
  5. Valida critérios de sucesso
  6. Salva modelo + metadados em model/artifacts/

REGRAS:
  ❌ Test set (mai/2024) NÃO é tocado aqui
  ❌ OGII não é calculado aqui — apenas em model/predict.py
  ❌ Thresholds do IPO não são recalculados — lidos de sprint0/thresholds.json
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
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
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

ARTIFACT_DIR = ROOT / "model" / "artifacts"
THRESH_FILE  = ROOT / "sprint0" / "thresholds.json"

# ─── Parâmetros ───────────────────────────────────────────────────────────────

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

SUCCESS_CRITERIA = {
    "min_f1_macro":        0.55,
    "min_recall_critical": 0.60,
}


# ─── Carga do banco ───────────────────────────────────────────────────────────

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
        "collected_at":      r.collected_at,
        "kp":                r.kp,
        "bz_nT":             r.bz_nT,
        "dst":               r.dst,
        "ae_index":          r.ae_index,
        "solar_wind_speed":  r.solar_wind_speed,
    } for r in rows])


# ─── Baseline heurístico ──────────────────────────────────────────────────────

def predict_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Regras simples baseadas em Kp e Bz.
    O XGBoost precisa superar isso para ser aceito.
    """
    preds = np.full(len(df), 3, dtype=int)
    preds[((df["kp"] < 7.0) & (df["bz_nT"] >= -10.0)).values] = 2
    preds[((df["kp"] < 5.0) & (df["bz_nT"] >= -5.0 )).values] = 1
    preds[((df["kp"] < 3.0) & (df["bz_nT"] >= 0.0  )).values] = 0
    return preds


# ─── Avaliação ────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    f1_macro  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report    = classification_report(y_true, y_pred, output_dict=True,
                                      zero_division=0)
    recall_c3 = report.get("3", {}).get("recall", 0.0)

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  F1-macro:          {f1_macro:.4f}")
    print(f"  Recall classe 3:   {recall_c3:.4f}")
    print()
    names = ["baixo", "moderado", "alto", "crítico"]
    for i in range(4):
        r = report.get(str(i), {})
        print(f"  Classe {i} ({names[i]:9s}):  "
              f"P={r.get('precision',0):.3f}  "
              f"R={r.get('recall',0):.3f}  "
              f"F1={r.get('f1-score',0):.3f}  "
              f"n={int(r.get('support',0))}")

    return {"f1_macro": f1_macro, "recall_critical": recall_c3,
            "report": report}


# ─── Gráficos ─────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, title: str, path: Path):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    lbls   = ["Baixo", "Moderado", "Alto", "Crítico"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_pct, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=lbls, yticklabels=lbls, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title, pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"  → {path.name}")


def save_feature_importance(model, feature_cols: list, path: Path):
    imp = model.feature_importances_
    df  = (pd.DataFrame({"feature": feature_cols, "importance": imp})
             .sort_values("importance", ascending=True))
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(df)))
    bars   = ax.barh(df["feature"], df["importance"], color=colors)
    for bar, v in zip(bars, df["importance"]):
        ax.text(bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)
    ax.set_title("Feature Importance — XGBoost (gain)", pad=10)
    ax.set_xlabel("Importância")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"  → {path.name}")


# ─── Pipeline de treino ───────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 52)
    print("  OrbitalShield — Treinamento do Modelo")
    print("=" * 52)

    # ── Thresholds congelados do Sprint 0 ────────────────────────────────────
    if not THRESH_FILE.exists():
        print(f"\nERRO: {THRESH_FILE} não encontrado.")
        print("Execute sprint0/01_ipo_distribution.py primeiro.")
        sys.exit(1)

    thresh_data = json.loads(THRESH_FILE.read_text())
    thresholds  = IPOThresholds(
        p25=thresh_data["p25"],
        p50=thresh_data["p50"],
        p75=thresh_data["p75"],
    )
    print(f"\nThresholds: p25={thresholds.p25}  "
          f"p50={thresholds.p50}  p75={thresholds.p75}")

    # ── Carga ─────────────────────────────────────────────────────────────────
    print("\nCarregando dados do banco...")
    raw_train = load_period("2018-01-01", "2023-12-31")
    raw_val   = load_period("2024-01-01", "2024-04-30")
    print(f"  Train bruto: {len(raw_train):,} | Val bruto: {len(raw_val):,}")

    if raw_train.empty:
        print("ERRO: train set vazio.")
        sys.exit(1)

    # ── Feature engineering ───────────────────────────────────────────────────
    print("Aplicando feature engineering...")
    train_df = build_features(raw_train, thresholds=thresholds)
    val_df   = build_features(raw_val,   thresholds=thresholds)

    train_df = train_df.dropna(subset=["ipo_future"] + FEATURE_COLS)
    val_df   = val_df.dropna(subset=["ipo_future"]   + FEATURE_COLS)

    missing = set(FEATURE_COLS) - set(train_df.columns)
    if missing:
        print(f"ERRO: features faltando: {missing}")
        sys.exit(1)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["ipo_future"].values.astype(int)
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df["ipo_future"].values.astype(int)

    print(f"  Train: {X_train.shape}  "
          f"classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Val:   {X_val.shape}    "
          f"classes: {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # ── Baseline ──────────────────────────────────────────────────────────────
    y_base    = predict_baseline(val_df)
    base_res  = evaluate(y_val, y_base, "Baseline Heurístico (Val)")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\nTreinando XGBoost...")
    model = xgb.XGBClassifier(**XGBOOST_PARAMS, early_stopping_rounds=EARLY_STOPPING)
    model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
    print(f"  Melhor iteração: {model.best_iteration}")
    print(f"  Melhor score:    {model.best_score:.6f}")

    y_pred    = model.predict(X_val)
    model_res = evaluate(y_val, y_pred, "XGBoost (Val)")

    # ── Critérios de sucesso ──────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  CRITÉRIOS DE SUCESSO")
    print("=" * 52)

    ok_f1     = model_res["f1_macro"]        >= SUCCESS_CRITERIA["min_f1_macro"]
    ok_beat   = model_res["f1_macro"]        >  base_res["f1_macro"]
    ok_recall = model_res["recall_critical"] >= SUCCESS_CRITERIA["min_recall_critical"]

    print(f"  F1-macro >= {SUCCESS_CRITERIA['min_f1_macro']}:              "
          f"{model_res['f1_macro']:.4f}  {'✅' if ok_f1 else '❌'}")
    print(f"  F1 > baseline ({base_res['f1_macro']:.4f}):        "
          f"{model_res['f1_macro']:.4f}  {'✅' if ok_beat else '❌'}")
    print(f"  Recall classe 3 >= {SUCCESS_CRITERIA['min_recall_critical']}:       "
          f"{model_res['recall_critical']:.4f}  {'✅' if ok_recall else '❌'}")

    all_ok = ok_f1 and ok_beat and ok_recall

    if not all_ok:
        print("\n❌ Modelo não passou nos critérios.")
        if not ok_beat:
            print("   → Modelo pior que regras simples — revisar features")
        if not ok_recall:
            print("   → Recall classe 3 baixo — classe rara no val set (só 36h de Kp>=7)")
        sys.exit(1)

    # ── Salva artefatos ───────────────────────────────────────────────────────
    print("\n✅ Modelo aprovado. Salvando artefatos...")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, ARTIFACT_DIR / "xgboost_model.joblib")

    metadata = {
        "feature_cols":   FEATURE_COLS,
        "label_col":      "ipo_future",
        "ipo_thresholds": {"p25": thresholds.p25, "p50": thresholds.p50,
                           "p75": thresholds.p75},
        "best_iteration": int(model.best_iteration),
        "best_score":     float(model.best_score),
        "xgboost_params": {k: v for k, v in XGBOOST_PARAMS.items()
                           if k not in ("objective", "eval_metric",
                                        "use_label_encoder", "verbosity")},
        "val_metrics":    {"f1_macro":        round(model_res["f1_macro"], 4),
                           "recall_critical": round(model_res["recall_critical"], 4)},
        "baseline_metrics": {"f1_macro": round(base_res["f1_macro"], 4)},
        "train_rows":     int(len(train_df)),
        "val_rows":       int(len(val_df)),
        "train_period":   "2018-01-01 → 2023-12-31",
        "val_period":     "2024-01-01 → 2024-04-30",
        "test_period":    "2024-05-01 → 2024-05-31  [NÃO USADO AQUI]",
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    (ARTIFACT_DIR / "ipo_thresholds.json").write_text(
        json.dumps({"p25": thresholds.p25, "p50": thresholds.p50,
                    "p75": thresholds.p75}, indent=2)
    )

    print("Gerando gráficos...")
    save_confusion_matrix(y_val, y_pred,
        "Matriz de Confusão — XGBoost (Val jan–abr 2024)",
        ARTIFACT_DIR / "cm_xgboost_val.png")
    save_confusion_matrix(y_val, y_base,
        "Matriz de Confusão — Baseline Heurístico (Val)",
        ARTIFACT_DIR / "cm_baseline_val.png")
    save_feature_importance(model, FEATURE_COLS,
        ARTIFACT_DIR / "feature_importance.png")

    print(f"\n  Artefatos salvos em: {ARTIFACT_DIR}/")
    print("  Próximo passo: python model/predict.py")
    print("=" * 52)


if __name__ == "__main__":
    train()