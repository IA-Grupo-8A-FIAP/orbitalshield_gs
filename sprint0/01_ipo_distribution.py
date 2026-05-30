# sprint0/01_ipo_distribution.py
"""
OrbitalShield — Sprint 0: Gate Científico
==========================================
Executa ANTES de qualquer treino. Se falhar: NÃO treinar.

Checks:
  1. Distribuição de classes — nenhuma classe < 5%
  2. Correlação entre componentes — alerta se |r| > 0.85
  3. Eventos extremos — >= 70% dos Kp>=7 devem ter IPO classe >= 2

Se aprovado: thresholds são salvos em sprint0/thresholds.json
Esses thresholds são lidos por model/train.py e model/predict.py.

Uso:
  python sprint0/01_ipo_distribution.py
"""

import json
import sys
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import SessionLocal
from db.models import SpaceWeatherRaw
from features.engineering import build_features
from features.ipo import (
    IPOThresholds,
    compute_ipo_components_vec,
    compute_ipo_score_vec,
    compute_ipo_class_vec,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR  = ROOT / "data" / "reports"
THRESH_FILE = ROOT / "sprint0" / "thresholds.json"


# ─── Carga ────────────────────────────────────────────────────────────────────

def load_train_data() -> pd.DataFrame:
    """Carrega 2018–2023 do banco (apenas source=omniweb)."""
    session = SessionLocal()
    try:
        rows = (
            session.query(SpaceWeatherRaw)
            .filter(
                SpaceWeatherRaw.source == "omniweb",
                SpaceWeatherRaw.collected_at >= "2018-01-01",
                SpaceWeatherRaw.collected_at <= "2023-12-31",
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

    logger.info(f"Carregados {len(df):,} registros (2018–2023)")
    return df


# ─── Preparação das features ──────────────────────────────────────────────────

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica build_features (southward_duration + ffill) e calcula
    componentes do IPO de forma vetorizada.
    """
    # build_features adiciona southward_duration e faz ffill
    df = build_features(df)

    # Garante colunas necessárias
    required = {"kp", "bz_nT", "southward_duration", "dst", "ae_index"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas faltando após build_features: {missing}")

    df = df.dropna(subset=list(required))
    logger.info(f"Após dropna: {len(df):,} registros")

    # Componentes + score vetorizados
    df = compute_ipo_components_vec(df)
    df["ipo_score"] = compute_ipo_score_vec(df).values

    return df


# ─── Check 1: Distribuição de Classes ─────────────────────────────────────────

def check1_distribution(df: pd.DataFrame):
    print("\n" + "=" * 52)
    print("CHECK 1 — Distribuição de Classes do IPO")
    print("=" * 52)

    # Calcula thresholds por percentil do train set
    p25 = float(np.percentile(df["ipo_score"], 25))
    p50 = float(np.percentile(df["ipo_score"], 50))
    p75 = float(np.percentile(df["ipo_score"], 75))
    thresholds = IPOThresholds(p25=round(p25, 4), p50=round(p50, 4), p75=round(p75, 4))

    print(f"\nThresholds calculados:")
    print(f"  p25 = {thresholds.p25}")
    print(f"  p50 = {thresholds.p50}")
    print(f"  p75 = {thresholds.p75}")

    df["ipo_class"] = compute_ipo_class_vec(df["ipo_score"], thresholds).values
    dist   = df["ipo_class"].value_counts(normalize=True).sort_index()
    counts = df["ipo_class"].value_counts().sort_index()
    names  = {0: "baixo", 1: "moderado", 2: "alto", 3: "crítico"}

    print(f"\nDistribuição de classes:")
    passed = True
    for cls in [0, 1, 2, 3]:
        pct   = float(dist.get(cls, 0.0))
        count = int(counts.get(cls, 0))
        flag  = "  ⚠️  RARO — revisar thresholds" if pct < 0.05 else "  ✅"
        print(f"  Classe {cls} ({names[cls]:9s}): {pct:.1%}  ({count:,}){flag}")
        if pct < 0.05:
            passed = False

    conclusion = "✅ APROVADO" if passed else "❌ REPROVADO — ajustar thresholds"
    print(f"\n  → {conclusion}")

    # Gráfico
    _plot_distribution(dist, names, thresholds)

    return passed, thresholds, df


def _plot_distribution(dist, names, thresholds):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
    vals   = [dist.get(i, 0) for i in range(4)]
    lbls   = [f"Classe {i}\n({names[i]})\n{vals[i]:.1%}" for i in range(4)]
    ax.bar(lbls, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.2, label="Mínimo (5%)")
    ax.set_title(
        f"Sprint 0 — Distribuição IPO (train 2018–2023)\n"
        f"p25={thresholds.p25}  p50={thresholds.p50}  p75={thresholds.p75}",
        pad=10
    )
    ax.set_ylabel("Proporção")
    ax.set_ylim(0, max(vals) * 1.30)
    ax.legend()
    plt.tight_layout()
    out = REPORT_DIR / "check1_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Gráfico salvo: {out}")


# ─── Check 2: Correlação entre Componentes ────────────────────────────────────

def check2_correlation(df: pd.DataFrame):
    print("\n" + "=" * 52)
    print("CHECK 2 — Correlação entre Componentes do IPO")
    print("=" * 52)

    pairs   = [("c1", "c2"), ("c1", "c3"), ("c2", "c3")]
    results = {}
    alert   = False

    for a, b in pairs:
        r, _ = pearsonr(df[a].dropna(), df[b].dropna())
        high  = abs(r) > 0.85
        flag  = "  ⚠️  REDUNDÂNCIA — revisar pesos" if high else "  ✅"
        print(f"  corr({a}, {b}) = {r:+.3f}{flag}")
        results[f"{a}x{b}"] = round(r, 4)
        if high:
            alert = True

    conclusion = "⚠️  ALERTA (não bloqueia gate)" if alert else "✅ OK"
    print(f"\n  → {conclusion}")

    # Heatmap
    _plot_correlation(df)

    # Check 2 é alerta, não bloqueia
    return (not alert), results


def _plot_correlation(df: pd.DataFrame):
    import seaborn as sns
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    corr = df[["c1", "c2", "c3"]].rename(columns={
        "c1": "C1 (Kp)", "c2": "C2 (Bz×h)", "c3": "C3 (Dst+AE)"
    }).corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=-1, vmax=1, center=0, square=True, ax=ax,
                annot_kws={"size": 12})
    ax.set_title("Sprint 0 — Correlação entre Componentes IPO", pad=10)
    plt.tight_layout()
    out = REPORT_DIR / "check2_correlation.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Gráfico salvo: {out}")


# ─── Check 3: Sensibilidade a Eventos Extremos ────────────────────────────────

def check3_extremes(df: pd.DataFrame):
    print("\n" + "=" * 52)
    print("CHECK 3 — Sensibilidade a Eventos Extremos (Kp >= 7)")
    print("=" * 52)

    eventos = df[df["kp"] >= 7.0]
    n_evt   = len(eventos)

    if n_evt == 0:
        # Não deveria acontecer com 52k registros de 2018-2023
        print("  ⚠️  Nenhum evento com Kp >= 7 encontrado no train set.")
        print("      Verifique se o Kp está na escala correta (0–9).")
        print("      Kp máximo encontrado:", df["kp"].max())
        return False, {}

    pct_alto = float((eventos["ipo_class"] >= 2).mean())
    ok       = pct_alto >= 0.70
    flag     = "✅" if ok else "❌  ABAIXO DE 70%"

    print(f"  Eventos com Kp >= 7:          {n_evt:,} horas")
    print(f"  % com IPO classe >= 2 (alto): {pct_alto:.1%}  {flag}")
    print(f"  Kp máximo no dataset:         {df['kp'].max():.2f}")

    # Distribuição do IPO nos eventos extremos
    ext_dist = eventos["ipo_class"].value_counts(normalize=True).sort_index()
    print(f"\n  IPO nos eventos Kp >= 7:")
    names = {0: "baixo", 1: "moderado", 2: "alto", 3: "crítico"}
    for cls, pct in ext_dist.items():
        print(f"    Classe {cls} ({names[cls]:9s}): {pct:.1%}")

    conclusion = "✅ APROVADO" if ok else "❌ REPROVADO"
    print(f"\n  → {conclusion}")

    # Gráfico
    _plot_extremes(df)

    return ok, {"n_events": n_evt, "pct_class_2_or_3": pct_alto}


def _plot_extremes(df: pd.DataFrame):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sample = df.sample(min(len(df), 8000), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 4))
    sc = ax.scatter(
        sample["kp"], sample["ipo_score"],
        c=sample["ipo_score"], cmap="RdYlGn_r",
        alpha=0.20, s=5, vmin=0, vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="IPO Score")
    ax.axhline(sample["ipo_score"].quantile(0.75),
               color="red",  linestyle="--", linewidth=1.1, label="p75 (limiar classe 3)")
    ax.axvline(7.0, color="gray", linestyle=":",  linewidth=1.0, label="Kp = 7")
    ax.set_xlabel("Kp (escala 0–9)")
    ax.set_ylabel("IPO Score")
    ax.set_xlim(-0.2, 9.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Sprint 0 — IPO Score vs Kp (amostra do train set)", pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = REPORT_DIR / "check3_extremes.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Gráfico salvo: {out}")


# ─── Gate Principal ────────────────────────────────────────────────────────────

def run_sprint0():
    print("=" * 52)
    print("  OrbitalShield — Sprint 0 Gate Científico")
    print("=" * 52)

    df = load_train_data()
    if df.empty:
        print("ERRO: banco vazio. Execute ingestion/omniweb_loader.py primeiro.")
        return False

    df = prepare(df)

    r1, thresholds, df = check1_distribution(df)
    r2, _              = check2_correlation(df)
    r3, _              = check3_extremes(df)

    # Check 2 é alerta, não bloqueia
    gate = r1 and r3

    print("\n" + "=" * 52)
    print("  RESULTADO FINAL")
    print("=" * 52)
    print(f"  Check 1 (Distribuição): {'✅ PASS' if r1 else '❌ FAIL'}")
    print(f"  Check 2 (Correlação):   {'✅ PASS' if r2 else '⚠️  ALERTA'}")
    print(f"  Check 3 (Extremos):     {'✅ PASS' if r3 else '❌ FAIL'}")
    print(f"\n  GATE: {'✅ APROVADO — pode treinar o modelo' if gate else '❌ REPROVADO — corrigir antes de treinar'}")

    if gate:
        # Salva thresholds para uso em train.py e predict.py
        THRESH_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "p25": thresholds.p25,
            "p50": thresholds.p50,
            "p75": thresholds.p75,
        }
        THRESH_FILE.write_text(json.dumps(payload, indent=2))
        print(f"\n  Thresholds salvos: {THRESH_FILE}")
        print("  Próximo passo: python model/train.py")
    else:
        print("\n  Ações necessárias antes de treinar:")
        if not r1:
            print("  → Ajustar denominadores em features/ipo.py (C2 ou C3 saturando)")
        if not r3:
            print("  → Verificar escala do Kp no banco (deve ser 0–9, não 0–90)")

    print()
    return gate


if __name__ == "__main__":
    ok = run_sprint0()
    sys.exit(0 if ok else 1)