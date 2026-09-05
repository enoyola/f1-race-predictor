from datetime import date

from f1_predictor.cache import DataCache
from f1_predictor.models import Weather
from f1_predictor.weather import WeatherClient
from tests.conftest import make_race


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_weather_is_wet_rules():
    assert Weather("d", 2.0, None, 20.0, "archive").is_wet
    assert not Weather("d", 0.2, None, 20.0, "archive").is_wet
    assert Weather("d", 0.0, 70.0, 20.0, "forecast").is_wet
    assert "dry" in Weather("d", 0.0, 10.0, 25.0, "forecast").describe()


def test_archive_vs_forecast_selection(tmp_path, monkeypatch):
    client = WeatherClient(cache=DataCache(str(tmp_path)))
    client.RATE_LIMIT_DELAY = 0
    urls = []

    def fake_get(url, params=None, timeout=None):
        urls.append(url)
        return FakeResponse({"daily": {"time": [params.get("start_date", "2025-03-02")], "precipitation_sum": [3.0],
                                       "precipitation_probability_max": [80], "temperature_2m_max": [19.5]}})

    monkeypatch.setattr("f1_predictor.weather.requests.get", fake_get)
    past = make_race(2025, 1)  # 2025-03-01
    w = client.get_race_weather(past, today=date(2025, 6, 1))
    assert w.source == "archive" and w.precipitation_mm == 3.0 and w.is_wet
    assert "archive" in urls[-1]

    soon = make_race(2025, 1)
    w2 = client.get_race_weather(soon, today=date(2025, 2, 25))
    assert w2.source == "forecast" and w2.precipitation_probability == 80
    assert "forecast" in urls[-1]

    far = client.get_race_weather(make_race(2025, 1), today=date(2024, 1, 1))
    assert far is None

    # cached: no new request
    n = len(urls)
    client.get_race_weather(past, today=date(2025, 6, 1))
    assert len(urls) == n


def test_missing_coordinates_returns_none(tmp_path):
    race = make_race(2025, 1)
    race.circuit.latitude = None
    assert WeatherClient(cache=DataCache(str(tmp_path))).get_race_weather(race) is None
