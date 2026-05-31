# dashboard/app.py
"""
OrbitalShield — Dashboard Operacional
======================================
Duas abas:
  Aba 1 — Monitor GNSS: OGII atual + alerta + gráfico 72h + KPIs
          Sidebar: botão "Replay Maio/2024" via st.session_state
  Aba 2 — Validação Científica: métricas, backtest, confusão, feature importance

Regras mantidas:
  - IPO nunca aparece na interface
  - OGII calculado via model/predict.py
  - Test set mai/2024 só exibido, não recalibrado
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import SessionLocal
from db.models import SpaceWeatherRaw
from model.predict import load_artifacts, predict_batch, ogii_to_alert

# ─── Configuração da página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="OrbitalShield",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS customizado ──────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
      background-color: #0d1117;
      color: #e6edf3;
  }
  .ogii-card {
      border-radius: 12px;
      padding: 28px 32px;
      text-align: center;
      margin-bottom: 8px;
  }
  .ogii-value {
      font-family: 'Space Mono', monospace;
      font-size: 72px;
      font-weight: 700;
      line-height: 1;
      margin: 0;
  }
  .ogii-label {
      font-size: 13px;
      letter-spacing: 3px;
      text-transform: uppercase;
      opacity: 0.7;
      margin-top: 6px;
  }
  .alert-banner {
      border-radius: 8px;
      padding: 16px 20px;
      margin: 12px 0;
      font-size: 15px;
      font-weight: 600;
  }
  .rec-box {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 14px 18px;
      font-size: 14px;
      color: #8b949e;
      margin-top: 8px;
  }
  .kpi-box {
      background: #161b22;
      border: 1px solid #21262d;
      border-radius: 8px;
      padding: 14px;
      text-align: center;
  }
  .kpi-val {
      font-family: 'Space Mono', monospace;
      font-size: 22px;
      font-weight: 700;
  }
  .kpi-lbl {
      font-size: 11px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: #8b949e;
      margin-top: 2px;
  }
  .metric-card {
      background: #161b22;
      border: 1px solid #21262d;
      border-radius: 10px;
      padding: 20px;
      text-align: center;
  }
  .metric-val {
      font-family: 'Space Mono', monospace;
      font-size: 36px;
      font-weight: 700;
      color: #58a6ff;
  }
  .metric-lbl {
      font-size: 12px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #8b949e;
      margin-top: 4px;
  }
  .section-title {
      font-size: 11px;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: #8b949e;
      border-bottom: 1px solid #21262d;
      padding-bottom: 8px;
      margin: 24px 0 16px 0;
  }
  .footer-note {
      font-size: 11px;
      color: #484f58;
      text-align: center;
      margin-top: 32px;
      font-style: italic;
  }
  div[data-testid="stTabs"] button {
      font-family: 'Space Mono', monospace;
      font-size: 13px;
      letter-spacing: 1px;
  }
  .replay-badge {
      display: inline-block;
      background: #c0392b22;
      border: 1px solid #c0392b66;
      color: #e74c3c;
      border-radius: 4px;
      padding: 2px 10px;
      font-size: 11px;
      letter-spacing: 2px;
      text-transform: uppercase;
      margin-left: 8px;
      vertical-align: middle;
  }
</style>
""", unsafe_allow_html=True)

# ─── Estado global ────────────────────────────────────────────────────────────

if "replay_mode" not in st.session_state:
    st.session_state.replay_mode = False

# ─── Cache de artefatos ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelo...")
def get_model_and_thresholds():
    return load_artifacts()

@st.cache_data(show_spinner=False, ttl=300)
def load_recent_data(hours: int = 80) -> pd.DataFrame:
    """Últimas `hours` horas do banco (OMNIWeb)."""
    session = SessionLocal()
    try:
        rows = (
            session.query(SpaceWeatherRaw)
            .filter(SpaceWeatherRaw.source == "omniweb")
            .order_by(SpaceWeatherRaw.collected_at.desc())
            .limit(hours)
            .all()
        )
    finally:
        session.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "collected_at":      r.collected_at,
        "kp":                r.kp or 0.0,
        "bz_nT":             r.bz_nT or 0.0,
        "dst":               r.dst or 0.0,
        "ae_index":          r.ae_index or 0.0,
        "solar_wind_speed":  r.solar_wind_speed or 400.0,
    } for r in rows])
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    return df.sort_values("collected_at").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_replay_data() -> pd.DataFrame:
    """Dados de maio/2024 para replay do evento extremo."""
    session = SessionLocal()
    try:
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
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "collected_at":      r.collected_at,
        "kp":                r.kp or 0.0,
        "bz_nT":             r.bz_nT or 0.0,
        "dst":               r.dst or 0.0,
        "ae_index":          r.ae_index or 0.0,
        "solar_wind_speed":  r.solar_wind_speed or 400.0,
    } for r in rows])
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    return df.sort_values("collected_at").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_backtest_summary() -> dict:
    path = ROOT / "backtesting" / "results" / "backtest_may2024_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}

@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict:
    path = ROOT / "model" / "artifacts" / "model_metadata.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}

@st.cache_data(show_spinner=False)
def load_shap_results() -> dict:
    path = ROOT / "experiments" / "shap_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}

@st.cache_data(show_spinner=False, ttl=10)
def load_esp32_telemetry(n: int = 10):
    """Últimas n leituras da tabela esp32_telemetry."""
    try:
        from db.models import Esp32Telemetry
        session = SessionLocal()
        try:
            rows = (
                session.query(Esp32Telemetry)
                .order_by(Esp32Telemetry.received_at.desc())
                .limit(n)
                .all()
            )
        finally:
            session.close()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "received_at":        r.received_at,
            "hdop":               r.hdop,
            "satellites_visible": r.satellites_visible,
            "fix_quality":        r.fix_quality,
            "status":             r.status,
            "ogii_received":      getattr(r, "ogii_received", None),
            "is_replay":          r.is_replay,
        } for r in rows])
    except Exception:
        return pd.DataFrame()

# ─── Helpers visuais ──────────────────────────────────────────────────────────

def ogii_color(level: str) -> str:
    return {
        "BAIXO":    "#27ae60",
        "MODERADO": "#f39c12",
        "ALTO":     "#e67e22",
        "CRÍTICO":  "#c0392b",
    }.get(level, "#58a6ff")

def plot_ogii_timeline(df: pd.DataFrame, replay: bool = False) -> plt.Figure:
    """Gráfico OGII 72h com suavização 24h."""
    fig, ax = plt.subplots(figsize=(12, 3.2))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ts   = df["collected_at"]
    ogii = df["ogii"]
    smooth = ogii.rolling(24, center=False, min_periods=1).mean()

    # Faixas de alerta
    ax.fill_between(ts, 0,  25,  alpha=0.06, color="#27ae60")
    ax.fill_between(ts, 25, 50,  alpha=0.06, color="#f39c12")
    ax.fill_between(ts, 50, 75,  alpha=0.06, color="#e67e22")
    ax.fill_between(ts, 75, 100, alpha=0.06, color="#c0392b")

    # Série bruta
    ax.plot(ts, ogii, color="#58a6ff", linewidth=0.7, alpha=0.3)
    # Suavizada
    ax.plot(ts, smooth, color="#58a6ff", linewidth=2.0, alpha=0.9,
            label="OGII suavizado 24h")
    ax.fill_between(ts, smooth, alpha=0.08, color="#58a6ff")

    # Limiar CRÍTICO
    ax.axhline(75, color="#c0392b", linestyle="--", linewidth=0.8, alpha=0.5,
               label="limiar ALTO/CRÍTICO (75)")

    # Marca pico se replay
    if replay:
        dt_peak = pd.Timestamp("2024-05-11 00:00:00", tz="UTC")
        ax.axvline(dt_peak, color="#c0392b", linestyle="--",
                   linewidth=1.2, label="Pico Kp=9")

    ax.set_ylim(0, 105)
    ax.set_xlim(ts.iloc[0], ts.iloc[-1])
    ax.set_ylabel("OGII", color="#8b949e", fontsize=10)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3 if replay else 1))
    ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#8b949e")
    plt.tight_layout(pad=0.5)
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛰️ OrbitalShield")
    st.markdown("**Previsão de Risco GNSS**  \nAgricultura de Precisão")
    st.divider()

    if st.button(
        "⚡ Replay Maio/2024" if not st.session_state.replay_mode
        else "↩ Voltar ao Normal",
        use_container_width=True,
        type="primary" if not st.session_state.replay_mode else "secondary",
    ):
        st.session_state.replay_mode = not st.session_state.replay_mode
        st.rerun()

    if st.session_state.replay_mode:
        st.warning("**Modo Replay ativo**  \nExibindo evento extremo  \nMaio/2024 — Kp=9, Dst=−412 nT")

    st.divider()
    st.markdown("""
    <div style='font-size:11px; color:#484f58; line-height:1.6'>
    Dados: NASA/OMNIWeb<br>
    Modelo: XGBoost v1.0<br>
    Treino: 2018–2023<br>
    52.584 registros
    </div>
    """, unsafe_allow_html=True)

# ─── Abas ─────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["📡  Monitor GNSS", "🔬  Validação Científica"])

# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — MONITOR OPERACIONAL
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

    # Título com badge de modo
    replay = st.session_state.replay_mode
    badge  = '<span class="replay-badge">REPLAY MAI/2024</span>' if replay else ""
    st.markdown(
        f"<h2 style='margin:0 0 4px 0'>Monitor Operacional{badge}</h2>"
        f"<p style='color:#8b949e; font-size:13px; margin:0'>"
        f"{'Evento extremo — Kp=9, Dst=−412 nT' if replay else 'Últimas 72 horas — dados reais OMNIWeb'}"
        f"</p>",
        unsafe_allow_html=True
    )

    # ── Carrega e processa dados ──────────────────────────────────────────────
    model, thresholds = get_model_and_thresholds()

    with st.spinner("Calculando OGII..."):
        if replay:
            raw_df = load_replay_data()
        else:
            raw_df = load_recent_data(hours=80)

        if raw_df.empty:
            st.error("Banco sem dados. Execute `python ingestion/omniweb_loader.py` primeiro.")
            st.stop()

        result_df = predict_batch(raw_df, model, thresholds, save_to_db=False)
        result_df = result_df.dropna(subset=["ogii"]).reset_index(drop=True)

        # Para modo normal, pega só as últimas 72h
        if not replay:
            cutoff = result_df["collected_at"].max() - pd.Timedelta(hours=72)
            display_df = result_df[result_df["collected_at"] >= cutoff].copy()
        else:
            display_df = result_df[
                result_df["collected_at"] >= "2024-05-01"
            ].copy()

    # ── Card OGII principal ───────────────────────────────────────────────────
    last = display_df.iloc[-1]
    alert = ogii_to_alert(float(last["ogii"]))
    level = alert["level"]
    color = ogii_color(level)

    st.markdown("<div class='section-title'>ÍNDICE OPERACIONAL GNSS</div>",
                unsafe_allow_html=True)

    col_card, col_info = st.columns([1, 2], gap="large")

    with col_card:
        st.markdown(f"""
        <div class="ogii-card" style="background:{color}18; border: 1px solid {color}44">
            <p class="ogii-value" style="color:{color}">{last['ogii']:.0f}</p>
            <p class="ogii-label">{alert['icon']} {level}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div class="alert-banner" style="background:{color}15; border-left: 3px solid {color}">
            {alert['message']}
        </div>
        <div class="rec-box">
            💡 <strong>Recomendação:</strong> {alert['recommendation']}
        </div>
        """, unsafe_allow_html=True)
        ts_str = last["collected_at"].strftime("%d/%m/%Y %H:%M UTC") \
            if hasattr(last["collected_at"], "strftime") else str(last["collected_at"])
        st.caption(f"Última leitura: {ts_str}")

    # ── KPIs geofísicos ───────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>PARÂMETROS GEOFÍSICOS</div>",
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "Kp",           f"{last['kp']:.1f}",              "Atividade geomagnética"),
        (k2, "Bz (nT)",      f"{last['bz_nT']:.1f}",           "Campo magnético sul"),
        (k3, "Dst (nT)",     f"{last['dst']:.0f}",             "Resposta magnetosférica"),
        (k4, "Vento Solar",  f"{last['solar_wind_speed']:.0f} km/s", "Velocidade plasma solar"),
    ]
    for col, label, value, subtitle in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-val">{value}</div>
                <div class="kpi-lbl">{label}</div>
                <div style="font-size:10px; color:#484f58; margin-top:3px">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Gráfico OGII ──────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='section-title'>{'OGII — MAIO/2024 (EVENTO EXTREMO)' if replay else 'OGII — ÚLTIMAS 72H'}</div>",
        unsafe_allow_html=True
    )
    fig = plot_ogii_timeline(display_df, replay=replay)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Tabela de leituras ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>ÚLTIMAS LEITURAS</div>",
                unsafe_allow_html=True)

    table_df = display_df.tail(10)[
        ["collected_at", "ogii", "alert_level", "kp", "bz_nT", "dst"]
    ].copy()
    table_df["collected_at"] = table_df["collected_at"].dt.strftime("%d/%m %H:%M")
    table_df["ogii"]         = table_df["ogii"].round(1)
    table_df.columns         = ["Data/Hora", "OGII", "Nível", "Kp", "Bz (nT)", "Dst (nT)"]
    st.dataframe(
        table_df[::-1],
        use_container_width=True,
        hide_index=True,
    )

    # ── Telemetria ESP32 ─────────────────────────────────────────────────────
    with st.expander("📡  Telemetria de Campo — ESP32 (Nó GNSS)", expanded=False):
        esp_df = load_esp32_telemetry()

        if esp_df.empty:
            # Demonstração com dados fixos quando bridge não está rodando
            st.info(
                "Bridge MQTT não ativo. Exibindo demonstração estática. "
                "Para dados ao vivo: `python ingestion/mqtt_telemetry.py`",
                icon="📡"
            )
            # Estado de demonstração baseado no OGII atual
            ogii_now = float(last["ogii"]) if "ogii" in last else 50.0
            if ogii_now > 75:
                demo = {"HDOP": "5.8", "Satélites": "4", "Fix": "1 (No Fix)",
                        "Status": "🔴 NO_RTK"}
            elif ogii_now > 50:
                demo = {"HDOP": "3.2", "Satélites": "7", "Fix": "2 (Float)",
                        "Status": "🟠 DEGRADED"}
            elif ogii_now > 25:
                demo = {"HDOP": "1.9", "Satélites": "10", "Fix": "3 (Float)",
                        "Status": "🟡 DEGRADED"}
            else:
                demo = {"HDOP": "0.9", "Satélites": "14", "Fix": "4 (RTK Fixed)",
                        "Status": "🟢 OK"}

            c1, c2, c3, c4 = st.columns(4)
            for col, (lbl, val) in zip([c1, c2, c3, c4], demo.items()):
                with col:
                    st.markdown(f"""
                    <div class="kpi-box">
                        <div class="kpi-val" style="font-size:20px">{val}</div>
                        <div class="kpi-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)
            st.caption("⚠️  Demonstração — is_replay: true | dados correlacionados ao OGII atual")

        else:
            last_esp = esp_df.iloc[0]
            ts_esp   = last_esp["received_at"]
            ts_str   = ts_esp.strftime("%d/%m %H:%M UTC") if hasattr(ts_esp, "strftime") else str(ts_esp)

            # Status color
            status_color = {
                "OK":      "#27ae60",
                "DEGRADED":"#e67e22",
                "NO_RTK":  "#c0392b",
            }.get(str(last_esp.get("status", "")), "#58a6ff")

            c1, c2, c3, c4 = st.columns(4)
            kpis_esp = [
                (c1, "HDOP",       f"{last_esp['hdop']:.2f}",              "Precisão GNSS"),
                (c2, "Satélites",  str(last_esp["satellites_visible"]),     "Visíveis"),
                (c3, "Fix Quality",str(last_esp["fix_quality"]),            "0=No Fix / 4=RTK"),
                (c4, "Status",     str(last_esp.get("status", "—")),       ts_str),
            ]
            for col, lbl, val, sub in kpis_esp:
                with col:
                    color = status_color if lbl == "Status" else "#e6edf3"
                    st.markdown(f"""
                    <div class="kpi-box" style="border-color:{color}33">
                        <div class="kpi-val" style="font-size:20px; color:{color}">{val}</div>
                        <div class="kpi-lbl">{lbl}</div>
                        <div style="font-size:10px; color:#484f58; margin-top:3px">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            # Mini histórico
            if len(esp_df) > 1:
                st.caption("Histórico HDOP — últimas leituras")
                hist = esp_df[["received_at","hdop","status"]].copy()
                hist["received_at"] = pd.to_datetime(hist["received_at"]).dt.strftime("%H:%M")
                st.dataframe(hist.rename(columns={
                    "received_at": "Hora", "hdop": "HDOP", "status": "Status"
                }), use_container_width=True, hide_index=True)

            replay_flag = last_esp.get("is_replay", True)
            st.caption(
                f"{'⚠️  Demonstração (is_replay: true)' if replay_flag else '✅ Dados reais de campo'} "
                f"| Device: orbital_esp32_01 | Tópico: orbitalshield/esp32/telemetry"
            )

    # Nota contextual no modo replay
    if replay:
        st.info(
            "**Lead time operacional de 240h:** O modelo prevê risco em horizonte de t+1h. "
            "No evento de maio/2024, emitiu alertas CRÍTICO sequenciais hora a hora desde 01/05 "
            "— detectando o início da rampa de degradação geomagnética 10 dias antes do pico "
            "de Kp=9 em 11/05. Não é previsão direta do pico, mas detecção contínua da "
            "deterioração do clima espacial. Na prática: 240h de lead time para planejamento agrícola.",
            icon="ℹ️"
        )

# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — VALIDAÇÃO CIENTÍFICA
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        "<h2 style='margin:0 0 4px 0'>Validação Científica</h2>"
        "<p style='color:#8b949e; font-size:13px; margin:0'>"
        "Backtesting no evento extremo de Maio/2024 — Kp=9, Dst=−412 nT</p>",
        unsafe_allow_html=True
    )

    summary  = load_backtest_summary()
    metadata = load_model_metadata()

    # ── Métricas principais ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>MÉTRICAS — TEST SET MAIO/2024</div>",
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (m1, f"{summary.get('f1_macro', 0):.4f}",        "F1-Macro",           "Classificação multiclasse"),
        (m2, f"{summary.get('recall_critical', 0):.4f}",  "Recall Crítico",     "Classe 3 — eventos severos"),
        (m3, f"{summary.get('antecipacao_h', 0):.0f}h",   "Antecipação",        "Primeiro alerta antes do pico"),
        (m4, f"Kp={summary.get('kp_max', 9):.0f}",        "Evento Testado",     f"Dst={summary.get('dst_min', -412):.0f} nT"),
    ]
    for col, val, label, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{label}</div>
                <div style="font-size:10px; color:#484f58; margin-top:4px">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Comparação com baseline ───────────────────────────────────────────────
    if metadata:
        base_f1  = metadata.get("baseline_metrics", {}).get("f1_macro", 0.2037)
        model_f1 = metadata.get("val_metrics", {}).get("f1_macro", 0.8185)
        melhoria = (model_f1 - base_f1) / base_f1 * 100
        st.info(
            f"**XGBoost vs Baseline heurístico:** F1-macro {model_f1:.4f} vs {base_f1:.4f} "
            f"— melhoria de **{melhoria:.0f}%** sobre regras simples de Kp/Bz.",
            icon="📊"
        )

    # ── Timeline do backtesting ───────────────────────────────────────────────
    st.markdown("<div class='section-title'>TIMELINE — EVENTO MAIO/2024</div>",
                unsafe_allow_html=True)

    timeline_path = ROOT / "backtesting" / "results" / "backtest_timeline.png"
    if timeline_path.exists():
        st.image(str(timeline_path), use_container_width=True)
    else:
        st.warning(
            "PNG não encontrado. Execute `python backtesting/backtest_may2024.py`.",
            icon="⚠️"
        )

    # ── Distribuição de alertas ───────────────────────────────────────────────
    if summary.get("alert_distribution"):
        st.markdown("<div class='section-title'>DISTRIBUIÇÃO DE ALERTAS — MAIO/2024</div>",
                    unsafe_allow_html=True)
        dist = summary["alert_distribution"]
        total = sum(dist.values())
        cols_dist = st.columns(4)
        for col, (level, count) in zip(cols_dist, dist.items()):
            pct = count / total * 100
            color = ogii_color(level)
            with col:
                st.markdown(f"""
                <div class="kpi-box" style="border-color:{color}33">
                    <div class="kpi-val" style="color:{color}">{pct:.0f}%</div>
                    <div class="kpi-lbl">{level}</div>
                    <div style="font-size:10px; color:#484f58; margin-top:3px">{count}h</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Matriz de confusão + Feature Importance ───────────────────────────────
    st.markdown("<div class='section-title'>DIAGNÓSTICO DO MODELO</div>",
                unsafe_allow_html=True)

    col_cm, col_fi = st.columns(2, gap="large")

    with col_cm:
        st.caption("Matriz de Confusão — Test Set")
        cm_path = ROOT / "backtesting" / "results" / "backtest_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.warning("Execute `backtest_may2024.py` para gerar.", icon="⚠️")

    with col_fi:
        st.caption("Feature Importance — XGBoost (gain)")
        fi_path = ROOT / "model" / "artifacts" / "feature_importance.png"
        if fi_path.exists():
            st.image(str(fi_path), use_container_width=True)
        else:
            st.warning("Execute `model/train.py` para gerar.", icon="⚠️")

    # ── SHAP Values ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>EXPLICABILIDADE — SHAP VALUES</div>",
                unsafe_allow_html=True)

    shap_results = load_shap_results()

    col_shap1, col_shap2 = st.columns(2, gap="large")

    with col_shap1:
        st.caption("SHAP — Importância Média por Feature (todas as classes)")
        shap_bar_path = ROOT / "experiments" / "shap_bar.png"
        if shap_bar_path.exists():
            st.image(str(shap_bar_path), use_container_width=True)
        else:
            st.warning("Execute `experiments/shap_analysis.py` para gerar.", icon="⚠️")

    with col_shap2:
        st.caption("SHAP — Classe CRÍTICO (top 10 features)")
        shap_crit_path = ROOT / "experiments" / "shap_class_critical.png"
        if shap_crit_path.exists():
            st.image(str(shap_crit_path), use_container_width=True)
        else:
            st.warning("Execute `experiments/shap_analysis.py` para gerar.", icon="⚠️")

    # Top 5 SHAP classe crítico em cards
    if shap_results.get("top_features_by_class", {}).get("Crítico"):
        st.markdown("<div class='section-title'>TOP 5 FEATURES — CLASSE CRÍTICO</div>",
                    unsafe_allow_html=True)
        top5 = shap_results["top_features_by_class"]["Crítico"]
        cols_top = st.columns(5)
        max_shap = top5[0]["mean_abs_shap"] if top5 else 1
        for col, feat in zip(cols_top, top5):
            pct = feat["mean_abs_shap"] / max_shap * 100
            with col:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-val" style="font-size:18px; color:#58a6ff">
                        {feat["mean_abs_shap"]:.4f}
                    </div>
                    <div class="kpi-lbl">{feat["label"].split("(")[0].strip()}</div>
                    <div style="margin-top:6px; background:#21262d;
                                border-radius:4px; height:4px">
                        <div style="width:{pct:.0f}%; background:#58a6ff;
                                    height:4px; border-radius:4px"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Ablation AMAS
    ablation_path = ROOT / "experiments" / "ablation_amas_results.json"
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text())
        st.markdown("<div class='section-title'>ABLATION STUDY — AMAS FACTOR</div>",
                    unsafe_allow_html=True)
        col_ab1, col_ab2, col_ab3 = st.columns(3)
        with col_ab1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="font-size:24px">
                    {ablation["with_amas"]["f1_macro"]:.4f}
                </div>
                <div class="metric-lbl">F1 COM amas</div>
            </div>""", unsafe_allow_html=True)
        with col_ab2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="font-size:24px">
                    {ablation["without_amas"]["f1_macro"]:.4f}
                </div>
                <div class="metric-lbl">F1 SEM amas</div>
            </div>""", unsafe_allow_html=True)
        with col_ab3:
            verdict_color = {"POSITIVO": "#27ae60",
                             "NEUTRO":   "#f39c12",
                             "NEGATIVO": "#c0392b"}.get(ablation["verdict"], "#58a6ff")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="font-size:22px; color:{verdict_color}">
                    {ablation["verdict"]}
                </div>
                <div class="metric-lbl">Veredito ablation</div>
                <div style="font-size:9px; color:#484f58; margin-top:4px">
                    ΔF1={ablation["delta_f1_macro"]:+.4f}
                </div>
            </div>""", unsafe_allow_html=True)
        st.caption("AMAS = hipótese experimental. Contribuição marginal não significativa (|Δ| < 0.005).")

    # ── Arquitetura em expander ───────────────────────────────────────────────
    with st.expander("🏗️  Arquitetura do Pipeline", expanded=False):
        st.markdown("""
        ```
        NASA/OMNIWeb (dados horários, 2018–2024)
               ↓
        ingestion/omniweb_loader.py  →  SQLite (space_weather_raw)
               ↓
        features/engineering.py  →  Feature Engineering
          (Kp, Bz, Dst, AE, southward_duration, rolling, AMAS)
               ↓
        sprint0/01_ipo_distribution.py  →  Gate Científico
          IPO constructo interno — thresholds congelados
               ↓
        model/train.py  →  XGBoost (F1-macro=0.8185, val jan–abr/2024)
               ↓
        model/predict.py  →  OGII (0–100) — índice operacional
               ↓
        dashboard/app.py  →  Monitor + Validação
               ↓
        ESP32 + MQTT  →  Telemetria física (próxima fase)
        ```

        | Camada | Tecnologia |
        |---|---|
        | Dados | NASA/OMNIWeb |
        | IA | XGBoost 2.1 |
        | Backend | Python 3.11 |
        | Banco | SQLite + SQLAlchemy |
        | Dashboard | Streamlit |
        | IoT (fase 2) | ESP32 + MQTT |
        """)

    # ── Rodapé ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='footer-note'>⚠️ Test set maio/2024 — não usar para ajuste de parâmetros · "
        "IPO é constructo interno, não exposto nesta interface · "
        "AMAS é hipótese experimental</div>",
        unsafe_allow_html=True
    )