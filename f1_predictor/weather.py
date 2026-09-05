"""
Race-day weather from Open-Meteo (free, no API key).

Historical race days use the archive endpoint; upcoming races within the
forecast horizon use the forecast endpoint. Anything further out is unknown.
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests

from f1_predictor.cache import DataCache
from f1_predictor.models import Race, Weather

logger = logging.getLogger(__name__)


class WeatherClient:
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    REQUEST_TIMEOUT = 10
    RATE_LIMIT_DELAY = 0.15
    FORECAST_HORIZON_DAYS = 15
    ARCHIVE_LAG_DAYS = 6  # archive data trails real time by a few days

    TTL_ARCHIVE = 365 * 24 * 3600
    TTL_FORECAST = 3 * 3600

    def __init__(self, cache: Optional[DataCache] = None, use_cache: bool = True):
        self.cache = cache if cache else DataCache()
        self.use_cache = use_cache
        self._last_request_time = 0.0
        self.request_count = 0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def get_race_weather(self, race: Race, today: Optional[date] = None) -> Optional[Weather]:
        """Return race-day weather, or None when unavailable."""
        circuit = race.circuit
        if circuit.latitude is None or circuit.longitude is None:
            return None

        today = today or date.today()
        race_day = race.date.date()
        day_str = race_day.isoformat()

        if race_day <= today - timedelta(days=self.ARCHIVE_LAG_DAYS):
            url, source, ttl = self.ARCHIVE_URL, "archive", self.TTL_ARCHIVE
            daily = "precipitation_sum,temperature_2m_max"
        elif race_day <= today + timedelta(days=self.FORECAST_HORIZON_DAYS):
            url, source, ttl = self.FORECAST_URL, "forecast", self.TTL_FORECAST
            daily = "precipitation_sum,precipitation_probability_max,temperature_2m_max"
        else:
            return None

        cache_key = f"weather_{race.season}_{race.round}_{source}"
        if self.use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return self._from_dict(cached)

        params = {
            "latitude": circuit.latitude,
            "longitude": circuit.longitude,
            "start_date": day_str,
            "end_date": day_str,
            "daily": daily,
            "timezone": "auto",
        }
        if source == "forecast":
            # The forecast endpoint only accepts past dates via past_days
            days_back = (today - race_day).days
            if days_back > 0:
                params.pop("start_date")
                params.pop("end_date")
                params["past_days"] = days_back
                params["forecast_days"] = 1

        try:
            self._rate_limit()
            self.request_count += 1
            response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as e:
            logger.debug(f"Weather unavailable for {race.race_name} {race.season}: {e}")
            return None

        weather = self._parse(payload, day_str, source)
        if weather is None:
            return None
        if self.use_cache:
            self.cache.set(cache_key, self._to_dict(weather), ttl)
        return weather

    @staticmethod
    def _pick(values, index) -> Optional[float]:
        try:
            v = values[index]
            return float(v) if v is not None else None
        except (IndexError, TypeError, ValueError):
            return None

    def _parse(self, payload: dict, day_str: str, source: str) -> Optional[Weather]:
        daily = payload.get("daily") or {}
        times = daily.get("time") or []
        if day_str not in times:
            return None
        idx = times.index(day_str)
        return Weather(
            date=day_str,
            precipitation_mm=self._pick(daily.get("precipitation_sum", []), idx),
            precipitation_probability=self._pick(daily.get("precipitation_probability_max", []), idx),
            temperature_max=self._pick(daily.get("temperature_2m_max", []), idx),
            source=source,
        )

    @staticmethod
    def _to_dict(w: Weather) -> dict:
        return {
            "date": w.date,
            "precipitation_mm": w.precipitation_mm,
            "precipitation_probability": w.precipitation_probability,
            "temperature_max": w.temperature_max,
            "source": w.source,
        }

    @staticmethod
    def _from_dict(d: dict) -> Weather:
        return Weather(
            date=d.get("date", ""),
            precipitation_mm=d.get("precipitation_mm"),
            precipitation_probability=d.get("precipitation_probability"),
            temperature_max=d.get("temperature_max"),
            source=d.get("source", "archive"),
        )
