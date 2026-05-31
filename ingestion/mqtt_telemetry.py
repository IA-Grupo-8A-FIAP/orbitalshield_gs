# ingestion/mqtt_telemetry.py
"""
OrbitalShield — MQTT Bridge: ESP32 → banco de dados
=====================================================
Assina orbitalshield/esp32/telemetry e persiste cada
payload recebido na tabela esp32_telemetry (SQLAlchemy).

Publica também orbitalshield/alerts a cada 30s com o
último OGII calculado — fecha o loop com o ESP32.

Uso:
    python ingestion/mqtt_telemetry.py

Pré-requisito:
    Broker MQTT acessível (padrão: test.mosquitto.org)
    Banco inicializado (python -c "from db.connection import init_db; init_db()")
"""

import json
import logging
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import SessionLocal
from db.models import Esp32Telemetry, RiskScore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Configuração ─────────────────────────────────────────────────────────────

# Credenciais via .env — nunca hardcoded em producao
from dotenv import load_dotenv
import os
load_dotenv()

MQTT_BROKER    = os.getenv("MQTT_BROKER",    "test.mosquitto.org")
MQTT_PORT      = int(os.getenv("MQTT_PORT",  "1883"))
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "orbitalshield_bridge_01")
MQTT_USERNAME  = os.getenv("MQTT_USERNAME")  # None = sem autenticacao
MQTT_PASSWORD  = os.getenv("MQTT_PASSWORD")  # None = sem autenticacao

TOPIC_TELEMETRY      = "orbitalshield/esp32/telemetry"
TOPIC_ALERTS         = "orbitalshield/alerts"

ALERT_PUBLISH_SEC    = 30    # intervalo de publicação do alerta para o ESP32


# ─── Persistência ─────────────────────────────────────────────────────────────

def save_telemetry(payload: dict):
    """Persiste um payload de telemetria na tabela esp32_telemetry."""
    session = SessionLocal()
    try:
        row = Esp32Telemetry(
            received_at        = datetime.now(timezone.utc),
            device_id          = payload.get("device_id", "orbital_esp32_01"),
            hdop               = float(payload.get("hdop", 0)),
            satellites_visible = int(payload.get("satellites_visible", 0)),
            satellites_used    = int(payload.get("satellites_used", 0)),
            fix_quality        = int(payload.get("fix_quality", 0)),
            latitude           = float(payload.get("latitude", 0)),
            longitude          = float(payload.get("longitude", 0)),
            status             = payload.get("status", "UNKNOWN"),
            is_replay          = bool(payload.get("is_replay", True)),
        )
        session.add(row)
        session.commit()
        logger.info(
            f"Telemetria salva | HDOP={row.hdop:.2f} "
            f"Sats={row.satellites_visible} Fix={row.fix_quality} "
            f"Status={row.status} Replay={row.is_replay}"
        )
    except Exception as e:
        session.rollback()
        logger.error(f"Erro ao salvar telemetria: {e}")
    finally:
        session.close()


def get_latest_ogii() -> dict:
    """
    Busca o OGII mais recente da tabela risk_scores.
    Retorna dict compatível com o payload do ESP32.
    """
    session = SessionLocal()
    try:
        row = (
            session.query(RiskScore)
            .order_by(RiskScore.scored_at.desc())
            .first()
        )
        if row:
            return {
                "ogii":  round(float(row.ogii), 1),
                "level": row.risk_label,
            }
    except Exception as e:
        logger.warning(f"Erro ao buscar OGII: {e}")
    finally:
        session.close()

    # Fallback se banco vazio
    return {"ogii": 50.0, "level": "MODERADO"}


# ─── Callbacks MQTT ───────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"Conectado ao broker {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(TOPIC_TELEMETRY)
        logger.info(f"Inscrito em: {TOPIC_TELEMETRY}")
    else:
        logger.error(f"Falha na conexão MQTT (rc={rc})")


def on_message(client, userdata, msg):
    topic   = msg.topic
    payload = msg.payload.decode("utf-8")

    logger.debug(f"Mensagem recebida [{topic}]: {payload}")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON inválido em {topic}: {e}")
        return

    if topic == TOPIC_TELEMETRY:
        save_telemetry(data)


def on_disconnect(client, userdata, rc):
    if rc != 0:
        logger.warning(f"Desconectado inesperadamente (rc={rc}). Reconectando...")


# ─── Publisher de alertas (loop paralelo) ─────────────────────────────────────

def alert_publisher_loop(client: mqtt.Client):
    """
    Thread paralela: publica o último OGII no tópico de alertas
    a cada ALERT_PUBLISH_SEC segundos para o ESP32 reagir.
    """
    logger.info(f"Publisher de alertas iniciado (intervalo: {ALERT_PUBLISH_SEC}s)")
    while True:
        try:
            alert = get_latest_ogii()
            payload = json.dumps(alert)
            result  = client.publish(TOPIC_ALERTS, payload)
            if result.rc == 0:
                logger.info(f"Alerta publicado: {payload}")
            else:
                logger.warning(f"Falha ao publicar alerta (rc={result.rc})")
        except Exception as e:
            logger.error(f"Erro no publisher de alertas: {e}")
        time.sleep(ALERT_PUBLISH_SEC)


# ─── Entry point ──────────────────────────────────────────────────────────────

def run():
    logger.info("=" * 50)
    logger.info("  OrbitalShield — MQTT Bridge")
    logger.info("  ESP32 Telemetry → SQLite")
    logger.info("=" * 50)

    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    # Autenticacao opcional via .env
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        logger.info("MQTT: autenticacao username/password ativada")
    else:
        logger.warning("MQTT: sem autenticacao — adequado apenas para demo/dev")

    logger.info(f"Conectando a {MQTT_BROKER}:{MQTT_PORT}...")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

    # Inicia publisher de alertas em thread paralela
    t = threading.Thread(target=alert_publisher_loop, args=(client,), daemon=True)
    t.start()

    logger.info("Bridge ativo. Aguardando telemetria do ESP32...")
    logger.info("Pressione Ctrl+C para encerrar.\n")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Encerrando bridge.")
        client.disconnect()


if __name__ == "__main__":
    run()