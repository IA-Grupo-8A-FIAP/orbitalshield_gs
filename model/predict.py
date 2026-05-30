# model/predict.py
"""
OrbitalShield — Inferência e Cálculo do OGII
=============================================
OGII = Operational GNSS Impact Index

Responsabilidades:
  1. Carregar modelo treinado de model/artifacts/
  2. Receber features e retornar probabilidades por classe IPO
  3. Calcular OGII a partir das probabilidades (NÃO das features)
  4. Converter OGII em alerta operacional para o agricultor
  5. Persistir resultado em risk_scores no banco

REGRAS CRÍTICAS:
  ❌ OGII não é calculado em nenhum outro módulo
  ❌ OGII não entra como feature — deriva das probabilidades
  ✅ Override físico documentado e logado (Kp >= 8)
  ✅ predict_single busca contexto histórico do banco (últimas 6h)
     para garantir que rolling features sejam calculadas corretamente

Uso direto:
  python model/predict.py
"""

import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import SessionLocal
from db.models import SpaceWeatherRaw, RiskScore
from features.engineering import build_features, FEATURE_COLS
from features.ipo import IPOThresholds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR  = ROOT / "model" / "artifacts"
THRESH_FILE   = ROOT / "sprint0" / "thresholds.json"
CONTEXT_HOURS = 12   # horas de histórico para calcular rolling features


# ─── Pesos do OGII ───────────────────────────────────────────────────────────

OGII_CLASS_WEIGHTS = {0: 0.0, 1: 30.0, 2: 65.0, 3: 100.0}
OGII_OVERRIDE_KP   = 8.0
OGII_OVERRIDE_MIN  = 80.0


# ─── Carregamento ────────────────────────────────────────────────────────────

def load_artifacts():
    """Carrega modelo e thresholds do disco."""
    model_path = ARTIFACT_DIR / "xgboost_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}\n"
            "Execute model/train.py primeiro."
        )
    model       = joblib.load(model_path)
    thresh_data = json.loads(THRESH_FILE.read_text())
    thresholds  = IPOThresholds(
        p25=thresh_data["p25"],
        p50=thresh_data["p50"],
        p75=thresh_data["p75"],
    )
    logger.info(f"Modelo carregado: {model_path.name}")
    return model, thresholds


# ─── OGII ────────────────────────────────────────────────────────────────────

def compute_ogii(
    proba: np.ndarray,
    kp_current: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calcula OGII a partir das probabilidades do modelo.
    OGII = Σ(prob_classe_i × peso_operacional_i), escala 0–100.
    """
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)

    weights = np.array([OGII_CLASS_WEIGHTS[i] for i in range(4)])
    ogii    = (proba * weights).sum(axis=1)

    if kp_current is not None:
        kp_arr  = np.atleast_1d(kp_current)
        extreme = kp_arr >= OGII_OVERRIDE_KP
        if extreme.any():
            logger.warning(
                f"Override físico: {extreme.sum()} instância(s) Kp >= "
                f"{OGII_OVERRIDE_KP} → OGII mínimo = {OGII_OVERRIDE_MIN}"
            )
            ogii = np.where(extreme, np.maximum(ogii, OGII_OVERRIDE_MIN), ogii)

    return np.clip(ogii, 0.0, 100.0)


def ogii_to_alert(ogii: float) -> dict:
    """Converte OGII em alerta operacional para o agricultor."""
    if ogii <= 25:
        return {
            "level":          "BAIXO",
            "color":          "#27ae60",
            "icon":           "🟢",
            "message":        "Condições normais — GNSS operacional",
            "recommendation": "Operação RTK normal. Precisão centimétrica esperada.",
        }
    elif ogii <= 50:
        return {
            "level":          "MODERADO",
            "color":          "#f39c12",
            "icon":           "🟡",
            "message":        "Atividade geomagnética moderada detectada",
            "recommendation": "Monitorar qualidade do sinal. Evitar operações RTK críticas nas próximas horas.",
        }
    elif ogii <= 75:
        return {
            "level":          "ALTO",
            "color":          "#e67e22",
            "icon":           "🟠",
            "message":        "Alta probabilidade de degradação GNSS",
            "recommendation": "Adiar operações de precisão centimétrica. Usar fallback DGNSS se disponível.",
        }
    else:
        return {
            "level":          "CRÍTICO",
            "color":          "#c0392b",
            "icon":           "🔴",
            "message":        "Degradação GNSS severa esperada",
            "recommendation": "Suspender operações RTK. Reagendar para próxima janela de baixa atividade.",
        }


# ─── Contexto histórico ───────────────────────────────────────────────────────

def _load_context_from_db(before_dt: datetime, hours: int = CONTEXT_HOURS) -> pd.DataFrame:
    """
    Busca as últimas `hours` horas do banco antes de before_dt.
    Necessário para calcular rolling features corretamente.
    """
    start = before_dt - timedelta(hours=hours)
    session = SessionLocal()
    try:
        rows = (
            session.query(SpaceWeatherRaw)
            .filter(
                SpaceWeatherRaw.source == "omniweb",
                SpaceWeatherRaw.collected_at >= start,
                SpaceWeatherRaw.collected_at <= before_dt,
            )
            .order_by(SpaceWeatherRaw.collected_at)
            .all()
        )
    finally:
        session.close()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([{
        "collected_at":      r.collected_at,
        "kp":                r.kp,
        "bz_nT":             r.bz_nT,
        "dst":               r.dst,
        "ae_index":          r.ae_index,
        "solar_wind_speed":  r.solar_wind_speed,
    } for r in rows])


# ─── Predição em batch ────────────────────────────────────────────────────────

def predict_batch(
    df: pd.DataFrame,
    model,
    thresholds: IPOThresholds,
    save_to_db: bool = False,
) -> pd.DataFrame:
    """
    Predição para um DataFrame de dados brutos (série temporal).
    O DataFrame deve ter linhas suficientes para rolling features (>= 6 linhas).
    """
    feat_df = build_features(df, thresholds=thresholds)
    feat_df = feat_df.dropna(subset=FEATURE_COLS)

    if feat_df.empty:
        logger.warning("Nenhuma linha válida para predição.")
        return feat_df

    X      = feat_df[FEATURE_COLS].values
    proba  = model.predict_proba(X)
    y_pred = model.predict(X)

    kp_vals = feat_df["kp"].values if "kp" in feat_df.columns else None
    ogii    = compute_ogii(proba, kp_current=kp_vals)

    result = feat_df.copy()
    result["ipo_class_pred"] = y_pred.astype(int)
    result["prob_0"] = proba[:, 0]
    result["prob_1"] = proba[:, 1]
    result["prob_2"] = proba[:, 2]
    result["prob_3"] = proba[:, 3]
    result["ogii"]        = ogii
    result["alert_level"] = [ogii_to_alert(v)["level"] for v in ogii]

    if save_to_db:
        _save_risk_scores(result)

    return result


def _save_risk_scores(result: pd.DataFrame):
    """Persiste predições na tabela risk_scores."""
    session = SessionLocal()
    try:
        for _, row in result.iterrows():
            alert = ogii_to_alert(float(row["ogii"]))
            score = RiskScore(
                scored_at        = row.get("collected_at",
                                           datetime.now(timezone.utc)),
                ogii             = float(row["ogii"]),
                risk_class       = int(row["ipo_class_pred"]),
                risk_label       = alert["level"],
                recommendation   = alert["recommendation"],
                dominant_feature = FEATURE_COLS[int(np.argmax(
                    np.abs(row[FEATURE_COLS].values)
                ))],
                model_version    = "v1.0",
            )
            session.add(score)
        session.commit()
        logger.info(f"Salvos {len(result)} registros em risk_scores.")
    except Exception as e:
        session.rollback()
        logger.error(f"Erro ao salvar risk_scores: {e}")
    finally:
        session.close()


# ─── Predição para um único instante (tempo real) ────────────────────────────

def predict_single(
    kp: float,
    bz_nT: float,
    dst: float,
    ae_index: float,
    solar_wind_speed: float = 400.0,
    reference_dt: Optional[datetime] = None,
    model=None,
    thresholds: IPOThresholds = None,
) -> dict:
    """
    Predição para um único instante com contexto histórico do banco.

    Busca as últimas CONTEXT_HOURS horas do banco para garantir que
    rolling features (kp_lag_3h, bz_min_3h, etc.) sejam calculadas
    corretamente — sem isso o modelo fica sem contexto temporal.

    Args:
        reference_dt: datetime de referência para buscar contexto.
                      Se None, usa datetime.now(UTC).
    """
    if model is None or thresholds is None:
        model, thresholds = load_artifacts()

    if reference_dt is None:
        reference_dt = datetime.now(timezone.utc)

    # Busca contexto histórico do banco
    context_df = _load_context_from_db(reference_dt, hours=CONTEXT_HOURS)

    # Monta linha atual
    now_row = pd.DataFrame([{
        "collected_at":      reference_dt,
        "kp":                kp,
        "bz_nT":             bz_nT,
        "dst":               dst,
        "ae_index":          ae_index,
        "solar_wind_speed":  solar_wind_speed,
    }])

    # Normaliza timezone do contexto (banco retorna naive, now_row é aware)
    if not context_df.empty:
        context_df["collected_at"] = pd.to_datetime(
            context_df["collected_at"]
        ).dt.tz_localize("UTC")
        full_df = pd.concat([context_df, now_row], ignore_index=True)
        logger.info(f"Contexto: {len(context_df)} linhas históricas + 1 atual")
    else:
        logger.warning(
            "Sem contexto histórico no banco para o instante solicitado. "
            "Rolling features serão zero — predição menos confiável."
        )
        full_df = now_row

    # Feature engineering na série completa
    feat_df = build_features(full_df, thresholds=thresholds)
    feat_df = feat_df.dropna(subset=FEATURE_COLS)

    if feat_df.empty:
        raise ValueError("Não foi possível calcular features para predição.")

    # Pega apenas a última linha (instante atual)
    last = feat_df.iloc[[-1]]
    X    = last[FEATURE_COLS].values

    proba = model.predict_proba(X)[0]
    pred  = int(model.predict(X)[0])
    ogii  = float(compute_ogii(
        proba.reshape(1, -1), kp_current=np.array([kp])
    )[0])
    alert = ogii_to_alert(ogii)

    return {
        "ipo_class_pred": pred,
        "probabilities": {
            "class_0_baixo":    round(float(proba[0]), 4),
            "class_1_moderado": round(float(proba[1]), 4),
            "class_2_alto":     round(float(proba[2]), 4),
            "class_3_critico":  round(float(proba[3]), 4),
        },
        "ogii":  round(ogii, 1),
        "alert": alert,
        "context_lines": len(context_df),
        "input": {
            "kp": kp, "bz_nT": bz_nT,
            "dst": dst, "ae_index": ae_index,
            "solar_wind_speed": solar_wind_speed,
        },
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, thresholds = load_artifacts()

    print("\n" + "=" * 52)
    print("  OrbitalShield — Exemplos de Predição")
    print("  (usando contexto histórico do banco)")
    print("=" * 52)

    # Usa datas dentro do banco (2024) para ter contexto real
    casos = [
    {
        "label":           "☀️  Dia quieto (Kp=1)",
        "kp":              1.0, "bz_nT": 2.0,
        "dst":             -5.0, "ae_index": 80.0,
        "solar_wind_speed": 380.0,
        "reference_dt":    datetime(2024, 2, 15, 12, 0, tzinfo=timezone.utc),
    },
    {
        "label":           "🟡 Atividade leve (Kp=3)",
        "kp":              3.0, "bz_nT": -4.0,
        "dst":             -15.0, "ae_index": 150.0,
        "solar_wind_speed": 430.0,
        "reference_dt":    datetime(2024, 2, 15, 14, 0, tzinfo=timezone.utc),
    },
    {
        "label":           "🟠 Tempestade moderada (Kp=6)",
        "kp":              6.0, "bz_nT": -12.0,
        "dst":             -80.0, "ae_index": 700.0,
        "solar_wind_speed": 600.0,
        "reference_dt":    datetime(2024, 4, 1, 6, 0, tzinfo=timezone.utc),
    },
    {
        "label":           "🔥 Evento extremo — mai/2024 (Kp=9)",
        "kp":              9.0, "bz_nT": -40.0,
        "dst":             -412.0, "ae_index": 2800.0,
        "solar_wind_speed": 900.0,
        "reference_dt":    datetime(2024, 5, 10, 22, 0, tzinfo=timezone.utc),
    },
]

    for caso in casos:
        label = caso.pop("label")
        result = predict_single(**caso, model=model, thresholds=thresholds)
        alert  = result["alert"]
        print(f"\n{label}")
        print(f"  Contexto histórico: {result['context_lines']} linhas")
        print(f"  IPO classe prevista: {result['ipo_class_pred']}")
        print(f"  Probabilidades:  "
              f"C0={result['probabilities']['class_0_baixo']:.3f}  "
              f"C1={result['probabilities']['class_1_moderado']:.3f}  "
              f"C2={result['probabilities']['class_2_alto']:.3f}  "
              f"C3={result['probabilities']['class_3_critico']:.3f}")
        print(f"  OGII: {result['ogii']:.1f}  {alert['icon']} {alert['level']}")
        print(f"  → {alert['recommendation']}")