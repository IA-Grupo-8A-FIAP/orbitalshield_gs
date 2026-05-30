# features/engineering.py
import numpy as np
import pandas as pd
from features.ipo import compute_ipo_components, compute_ipo_score, compute_ipo_class, IPOThresholds


def add_southward_duration(df: pd.DataFrame,
                            bz_col: str = "bz_nT",
                            threshold: float = -5.0,
                            interval_h: float = 1.0) -> pd.DataFrame:
    """
    Horas consecutivas com Bz abaixo do threshold.
    interval_h = 1.0 para dados horários do OMNIWeb.
    """
    duration = []
    count = 0.0
    for val in df[bz_col].fillna(0):
        count = (count + interval_h) if val < threshold else 0.0
        duration.append(round(count, 2))
    df["southward_duration"] = duration
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Janelas temporais com shift(1) obrigatório — evita leakage.
    Dados horários: shift(1)=1h, rolling(3)=3h, rolling(6)=6h.
    """
    kp_shifted = df["kp"].shift(1)
    df["kp_lag_1h"]  = df["kp"].shift(1)
    df["kp_lag_3h"]  = df["kp"].shift(3)
    df["kp_mean_3h"] = kp_shifted.rolling(3).mean()
    df["kp_mean_6h"] = kp_shifted.rolling(6).mean()

    bz_shifted = df["bz_nT"].shift(1)
    df["bz_min_3h"]  = bz_shifted.rolling(3).min()
    df["bz_mean_3h"] = bz_shifted.rolling(3).mean()

    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Kp x |Bz negativo| — captura eventos combinados.
    Clipa em 25.0 (aprox. p99 do train set 2018-2023) para evitar
    que valores extremos sintéticos ou fora da distribuição de treino
    dominem a predição.
    """
    raw = df["kp"] * df["bz_nT"].clip(upper=0).abs()
    df["kp_x_bz_neg"] = raw.clip(upper=25.0)
    return df


def add_cyclical_time(df: pd.DataFrame,
                       lon_ref: float = -47.0) -> pd.DataFrame:
    """
    Hora local solar codificada como sin/cos.
    Encoding cíclico garante continuidade entre hora 23 e 0.
    """
    df["hour_utc"]   = pd.to_datetime(df["collected_at"]).dt.hour
    df["hour_local"] = (df["hour_utc"] + lon_ref / 15.0) % 24
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour_local"] / 24)
    return df


def add_amas_factor(df: pd.DataFrame,
                     lat: float = -15.0,
                     lon: float = -50.0) -> pd.DataFrame:
    """
    Fator de ajuste regional baseado no IGRF-13.
    HIPÓTESE EXPERIMENTAL — validar via ablation study.
    f = 1 + 0.35 * (1 - B_local / B_referencia)
    """
    B_ref = 40000.0

    if -30 <= lat <= -5 and -60 <= lon <= -30:
        B_local = 24000.0 + (abs(lat + 17) / 12.0) * 4000.0
    elif -5 <= lat <= 5 and -50 <= lon <= -35:
        B_local = 32000.0
    else:
        B_local = B_ref

    factor = 1.0 + 0.35 * (1.0 - B_local / B_ref)
    df["amas_factor"] = round(max(1.0, min(1.35, factor)), 3)
    return df


def add_ipo_label(df: pd.DataFrame,
                   thresholds: IPOThresholds = None,
                   forecast_steps: int = 1) -> pd.DataFrame:
    """
    Calcula IPO para cada linha e cria label futuro (t+1h).
    forecast_steps=1 para dados horários = previsão 1h à frente.

    CRÍTICO: label é IPO futuro, não IPO atual.
    Features em t preveem IPO em t+1h.
    """
    def row_ipo(row):
        try:
            comps = compute_ipo_components(
                kp=row["kp"] or 0,
                bz=row["bz_nT"] or 0,
                southward_h=row["southward_duration"],
                dst=row["dst"] or 0,
                ae=row["ae_index"] or 0,
            )
            score = compute_ipo_score(**comps)
            return compute_ipo_class(score, thresholds)
        except Exception:
            return np.nan

    df["ipo_current"] = df.apply(row_ipo, axis=1)
    df["ipo_future"]  = df["ipo_current"].shift(-forecast_steps)
    return df


# Features usadas pelo modelo — ordem importa para SHAP
FEATURE_COLS = [
    # Essenciais
    "kp", "bz_nT", "bz_min_3h", "dst", "ae_index",
    "southward_duration", "hour_sin", "hour_cos",
    # Importantes
    "solar_wind_speed", "kp_lag_3h", "kp_mean_6h",
    "kp_x_bz_neg", "amas_factor",
]


def build_features(df: pd.DataFrame,
                    lat: float = -15.0,
                    lon: float = -50.0,
                    thresholds: IPOThresholds = None) -> pd.DataFrame:
    """Pipeline completo de feature engineering."""
    df = df.copy().sort_values("collected_at").reset_index(drop=True)
    df = df.ffill().fillna(0)

    df = add_southward_duration(df)
    df = add_rolling_features(df)
    df = add_interactions(df)
    df = add_cyclical_time(df, lon_ref=lon)
    df = add_amas_factor(df, lat=lat, lon=lon)
    df = add_ipo_label(df, thresholds=thresholds)

    return df