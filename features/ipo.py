# features/ipo.py
"""
OrbitalShield — Cálculo do IPO (Índice de Perturbação Operacional)
===================================================================
O IPO é um constructo interno para treino do modelo.
NÃO é variável física direta. NÃO é exposto ao usuário.
O usuário vê apenas o OGII, calculado em model/predict.py.

Três componentes independentes:
  C1 — Intensidade geomagnética global (Kp)
  C2 — Campo interplanetário sul persistente (Bz × duração)
  C3 — Resposta magnetosférica (Dst + AE)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class IPOThresholds:
    """
    Thresholds calculados no Sprint 0 a partir do train set (2018–2023).
    CONGELADOS após definição — não modificar depois do treino.
    """
    p25: float = 0.18
    p50: float = 0.31
    p75: float = 0.52


# ─── Versão escalar (para inferência em tempo real, 1 instante) ──────────────

def compute_ipo_components(
    kp: float,
    bz: float,
    southward_h: float,
    dst: float,
    ae: float,
) -> dict:
    """
    Calcula as três componentes do IPO para um único instante.
    Cada componente normalizada em [0, 1].

    C1 — Kp normalizado pela escala oficial (0–9)
    C2 — Produto Bz_sul × duração, com normalização relativa ao percentil 95
         dos eventos históricos (evita saturação prematura em eventos moderados)
    C3 — Média ponderada de Dst e AE normalizados
    """
    # C1: Kp / 9
    c1 = float(np.clip(kp / 9.0, 0.0, 1.0))

    # C2: bz_south × southward_h
    # Denominador 150: calibrado para que Bz=-10nT × 15h = 1.0
    # (evento moderado-severo prolongado ≈ referência de saturação)
    # Gonzalez et al. 1994 define tempestade intensa: Dst < -100nT após
    # Bz < -10nT por >= 3h. Usamos margem maior para não saturar cedo.
    bz_south = abs(min(float(bz), 0.0))
    c2 = float(np.clip(bz_south * float(southward_h) / 150.0, 0.0, 1.0))

    # C3: Dst e AE normalizados, Dst tem mais peso (0.6/0.4)
    # Dst ref 300 nT: cobre ≥ 95% dos eventos históricos sem saturar
    # AE ref 2000 nT: atividade auroral intensa
    dst_norm = float(np.clip(abs(min(float(dst), 0.0)) / 300.0, 0.0, 1.0))
    ae_norm  = float(np.clip(float(ae) / 2000.0, 0.0, 1.0))
    c3 = 0.60 * dst_norm + 0.40 * ae_norm

    return {"c1": round(c1, 4), "c2": round(c2, 4), "c3": round(c3, 4)}


def compute_ipo_score(c1: float, c2: float, c3: float) -> float:
    """
    Score IPO = média simples das três componentes.
    Pesos iguais por parcimônia — sem evidência empírica para
    ponderação diferencial neste estágio (a ser revisado após ablation).
    """
    return round((c1 + c2 + c3) / 3.0, 4)


def compute_ipo_class(ipo_score: float, thresholds: IPOThresholds = None) -> int:
    """
    Binariza o score em 4 classes usando thresholds do train set.
    Classe 0: baixo | 1: moderado | 2: alto | 3: crítico
    """
    if thresholds is None:
        thresholds = IPOThresholds()
    if ipo_score < thresholds.p25:
        return 0
    if ipo_score < thresholds.p50:
        return 1
    if ipo_score < thresholds.p75:
        return 2
    return 3


def compute_ipo(
    kp: float,
    bz: float,
    southward_h: float,
    dst: float,
    ae: float,
    thresholds: IPOThresholds = None,
) -> int:
    """Função principal escalar — retorna classe IPO 0–3."""
    comps = compute_ipo_components(kp, bz, southward_h, dst, ae)
    score = compute_ipo_score(**comps)
    return compute_ipo_class(score, thresholds)


# ─── Versão vetorizada (para Sprint 0 e treino — muito mais rápido) ──────────

def compute_ipo_components_vec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versão vetorizada de compute_ipo_components.
    Espera colunas: kp, bz_nT, southward_duration, dst, ae_index

    Retorna DataFrame com colunas c1, c2, c3 adicionadas.
    """
    out = df.copy()

    # C1
    out["c1"] = np.clip(out["kp"] / 9.0, 0.0, 1.0)

    # C2
    bz_south = (-out["bz_nT"]).clip(lower=0.0)
    out["c2"] = np.clip(bz_south * out["southward_duration"] / 150.0, 0.0, 1.0)

    # C3
    dst_norm = np.clip((-out["dst"]).clip(lower=0.0) / 300.0, 0.0, 1.0)
    ae_norm  = np.clip(out["ae_index"] / 2000.0, 0.0, 1.0)
    out["c3"] = 0.60 * dst_norm + 0.40 * ae_norm

    return out


def compute_ipo_score_vec(df: pd.DataFrame) -> pd.Series:
    """
    Versão vetorizada de compute_ipo_score.
    Espera colunas c1, c2, c3 no DataFrame.
    """
    return ((df["c1"] + df["c2"] + df["c3"]) / 3.0).rename("ipo_score")


def compute_ipo_class_vec(
    ipo_score: pd.Series,
    thresholds: IPOThresholds,
) -> pd.Series:
    """
    Versão vetorizada de compute_ipo_class.
    """
    classes = pd.Series(3, index=ipo_score.index, dtype=int)
    classes[ipo_score < thresholds.p75] = 2
    classes[ipo_score < thresholds.p50] = 1
    classes[ipo_score < thresholds.p25] = 0
    return classes.rename("ipo_class")