# ingestion/omniweb_loader.py
import requests
import logging
import pandas as pd
from datetime import datetime, timezone
from db.connection import SessionLocal
from db.models import SpaceWeatherRaw

logger = logging.getLogger(__name__)

OMNIWEB_URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# Mapeamento validado em 30/05/2026:
# 38=Kp, 40=Dst, 41=AE, 16=Bz(GSM), 24=SW Speed
OMNIWEB_VARS = ["38", "40", "41", "16", "24"]


def fetch_omniweb(start: str, end: str) -> str | None:
    """
    Busca dados históricos do OMNIWeb.
    start/end formato: YYYYMMDD
    Cada variável é um parâmetro separado — OMNIWeb não aceita vírgulas.
    """
    params = [
        ("activity", "retrieve"),
        ("res",       "hour"),
        ("spacecraft","omni2"),
        ("start_date", start),
        ("end_date",   end),
        ("scale",     "Linear"),
        ("view",      "0"),
        ("table",     "0"),
        ("email",     "none"),
    ]
    for v in OMNIWEB_VARS:
        params.append(("vars", v))

    try:
        r = requests.get(OMNIWEB_URL, params=params, timeout=60)
        r.raise_for_status()
        if "Error" in r.text[:200]:
            logger.error(f"OMNIWeb retornou erro: {r.text[:300]}")
            return None
        return r.text
    except Exception as e:
        logger.error(f"Erro OMNIWeb {start}-{end}: {e}")
        return None


def parse_omniweb(text: str) -> pd.DataFrame:
    """
    Parseia o texto ASCII do OMNIWeb.
    Formato: YEAR DOY HR  kp  dst  ae  bz  vsw
    """
    lines = []
    header_found = False

    for line in text.splitlines():
        line = line.strip()

        # Detecta linha de cabeçalho
        if line.startswith("YEAR"):
            header_found = True
            continue

        if not header_found:
            continue

        if not line or not line[0].isdigit():
            continue

        parts = line.split()
        if len(parts) < 8:
            continue

        try:
            year = int(parts[0])
            doy  = int(parts[1])
            hour = int(parts[2])
            kp   = float(parts[3])
            dst  = float(parts[4])
            ae   = float(parts[5])
            bz   = float(parts[6])
            vsw  = float(parts[7])

            dt = datetime(year, 1, 1, tzinfo=timezone.utc) + \
                 pd.Timedelta(days=doy - 1, hours=hour)

            # Valores ausentes do OMNIWeb: 999, 9999, 99999
            lines.append({
                "collected_at":     dt,
                "kp": round(kp / 10.0, 2) if kp < 900 else None,
                "dst":              dst if abs(dst) < 999 else None,
                "ae_index":         ae  if ae  < 9999  else None,
                "bz_nT":            bz  if abs(bz) < 999 else None,
                "solar_wind_speed": vsw if vsw < 9999  else None,
                "source":           "omniweb",
            })
        except Exception:
            continue

    return pd.DataFrame(lines)


def save_dataframe(df: pd.DataFrame) -> int:
    """
    Persiste registros no banco com deduplicação por (collected_at, source).
    Registros já existentes são ignorados — reexecutar o loader é seguro.
    Retorna o número de registros novos inseridos.
    """
    if df.empty:
        return 0

    session = SessionLocal()
    inserted = 0
    skipped  = 0

    try:
        for _, row in df.iterrows():
            # Verifica se já existe registro para esse timestamp + fonte
            exists = (
                session.query(SpaceWeatherRaw)
                .filter(
                    SpaceWeatherRaw.collected_at == row["collected_at"],
                    SpaceWeatherRaw.source       == row["source"],
                )
                .first()
            )
            if exists:
                skipped += 1
                continue

            obj = SpaceWeatherRaw(**row.to_dict())
            session.add(obj)
            inserted += 1

        session.commit()

        if skipped > 0:
            logger.info(f"  Deduplicação: {inserted} inseridos, {skipped} ignorados (já existiam)")

        return inserted

    except Exception as e:
        session.rollback()
        logger.error(f"Erro ao salvar: {e}")
        return 0
    finally:
        session.close()


def load_historical(start_year: int = 2018, end_year: int = 2023):
    """Carrega histórico anual do OMNIWeb, ano a ano."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    total = 0
    for year in range(start_year, end_year + 1):
        start = f"{year}0101"
        end   = f"{year}1231"
        logger.info(f"Baixando {year}...")

        text = fetch_omniweb(start, end)
        if not text:
            logger.warning(f"Sem dados para {year}.")
            continue

        df = parse_omniweb(text)
        if df.empty:
            logger.warning(f"Nenhum dado parseado para {year}.")
            continue

        saved = save_dataframe(df)
        total += saved
        logger.info(f"{year}: {saved} registros novos. Acumulado: {total}")

    logger.info(f"Carga histórica concluída. Total inserido: {total} registros.")


if __name__ == "__main__":
    load_historical()