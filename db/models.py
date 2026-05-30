# db/models.py
from sqlalchemy import (
    Column, Integer, Float, String,
    DateTime, Boolean
)
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class SpaceWeatherRaw(Base):
    """Dados brutos coletados das APIs externas."""
    __tablename__ = "space_weather_raw"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    collected_at     = Column(DateTime(timezone=True), nullable=False, index=True)
    source           = Column(String(50), nullable=False)
    kp               = Column(Float)
    bz_nT            = Column(Float)
    dst              = Column(Float)
    ae_index         = Column(Float)
    solar_wind_speed = Column(Float)
    is_interpolated  = Column(Boolean, default=False)
    created_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RiskScore(Base):
    """Resultado de cada inferência do modelo."""
    __tablename__ = "risk_scores"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    scored_at        = Column(DateTime(timezone=True), nullable=False, index=True)
    ogii             = Column(Float, nullable=False)
    risk_class       = Column(Integer, nullable=False)
    risk_label       = Column(String(20), nullable=False)
    recommendation   = Column(String(200))
    dominant_feature = Column(String(100))
    model_version    = Column(String(50), default="v1.0")


class Esp32Telemetry(Base):
    """Telemetria do sensor físico."""
    __tablename__ = "esp32_telemetry"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    received_at        = Column(DateTime(timezone=True), nullable=False, index=True)
    device_id          = Column(String(50), default="orbital_esp32_01")
    hdop               = Column(Float)
    satellites_visible = Column(Integer)
    satellites_used    = Column(Integer)
    fix_quality        = Column(Integer)
    latitude           = Column(Float)
    longitude          = Column(Float)
    status             = Column(String(20))
    is_replay          = Column(Boolean, default=False)