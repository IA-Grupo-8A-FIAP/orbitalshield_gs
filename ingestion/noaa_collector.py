# ingestion/noaa_collector.py
import requests
import logging
from datetime import datetime, timezone
from db.connection import SessionLocal
from db.models import SpaceWeatherRaw

logger = logging.getLogger(__name__)

ENDPOINTS = {
    "kp":     "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "mag":    "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
    "plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
}


def fetch(url: str) -> list | None:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Erro ao buscar {url}: {e}")
        return None


def parse_kp(data: list) -> list[dict]:
    records = []
    for row in data:
        try:
            # novo formato: dicionário com time_tag e Kp
            if isinstance(row, dict):
                records.append({
                    "collected_at": datetime.fromisoformat(row["time_tag"]).replace(tzinfo=timezone.utc),
                    "kp":           float(row["Kp"]),
                    "source":       "noaa_kp",
                })
            # formato antigo: lista
            elif isinstance(row, list) and len(row) >= 2:
                records.append({
                    "collected_at": datetime.fromisoformat(row[0]).replace(tzinfo=timezone.utc),
                    "kp":           float(row[1]),
                    "source":       "noaa_kp",
                })
        except Exception:
            continue
    return records


def parse_mag(data: list) -> list[dict]:
    records = []
    for row in data[1:] if isinstance(data[0], list) else data:
        try:
            if isinstance(row, dict):
                bz = row.get("bz_gsm") or row.get("Bz")
                records.append({
                    "collected_at": datetime.fromisoformat(row["time_tag"]).replace(tzinfo=timezone.utc),
                    "bz_nT":        float(bz) if bz not in (None, "") else None,
                    "source":       "noaa_mag",
                })
            elif isinstance(row, list) and len(row) >= 4:
                records.append({
                    "collected_at": datetime.fromisoformat(row[0]).replace(tzinfo=timezone.utc),
                    "bz_nT":        float(row[3]) if row[3] not in ("", "-9999.99") else None,
                    "source":       "noaa_mag",
                })
        except Exception:
            continue
    return records


def parse_plasma(data: list) -> list[dict]:
    records = []
    for row in data[1:] if isinstance(data[0], list) else data:
        try:
            if isinstance(row, dict):
                speed = row.get("speed") or row.get("bulk_speed")
                records.append({
                    "collected_at":    datetime.fromisoformat(row["time_tag"]).replace(tzinfo=timezone.utc),
                    "solar_wind_speed": float(speed) if speed not in (None, "") else None,
                    "source":          "noaa_plasma",
                })
            elif isinstance(row, list) and len(row) >= 3:
                records.append({
                    "collected_at":    datetime.fromisoformat(row[0]).replace(tzinfo=timezone.utc),
                    "solar_wind_speed": float(row[2]) if row[2] not in ("", "-9999.9") else None,
                    "source":          "noaa_plasma",
                })
        except Exception:
            continue
    return records


def save(records: list[dict]):
    if not records:
        return
    session = SessionLocal()
    try:
        for rec in records:
            obj = SpaceWeatherRaw(**rec)
            session.add(obj)
        session.commit()
        logger.info(f"{len(records)} registros salvos.")
    except Exception as e:
        session.rollback()
        logger.error(f"Erro ao salvar: {e}")
    finally:
        session.close()


def collect_all():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Iniciando coleta NOAA...")

    kp_data = fetch(ENDPOINTS["kp"])
    if kp_data:
        save(parse_kp(kp_data))

    mag_data = fetch(ENDPOINTS["mag"])
    if mag_data:
        save(parse_mag(mag_data))

    plasma_data = fetch(ENDPOINTS["plasma"])
    if plasma_data:
        save(parse_plasma(plasma_data))

    logger.info("Coleta concluída.")


if __name__ == "__main__":
    collect_all()