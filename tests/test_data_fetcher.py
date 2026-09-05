import pytest
import requests

from f1_predictor.cache import DataCache
from f1_predictor.data_fetcher import F1DataFetcher


def _race_payload(season, rnd, rows, key="Results", sprint=False):
    race = {
        "season": str(season), "round": str(rnd), "raceName": f"GP {rnd}",
        "Circuit": {"circuitId": f"c{rnd}", "circuitName": f"Circuit {rnd}", "Location": {"lat": "1.5", "long": "2.5", "locality": "Town", "country": "Land"}},
        "date": f"{season}-03-0{rnd}", "time": "14:00:00Z",
        key: rows,
    }
    if sprint:
        race["Sprint"] = {"date": f"{season}-03-0{rnd}"}
    return race


def _row(driver_id, pos, grid=None, status="Finished", points="25"):
    return {
        "Driver": {"driverId": driver_id, "code": driver_id[:3].upper(), "givenName": "A", "familyName": driver_id.title(), "nationality": "X"},
        "Constructor": {"constructorId": "team", "name": "Team", "nationality": "X"},
        "position": str(pos), "points": points, "grid": str(grid or pos), "laps": "50", "status": status,
    }


@pytest.fixture
def fetcher(tmp_path):
    f = F1DataFetcher(cache=DataCache(str(tmp_path)), use_cache=True)
    f.RATE_LIMIT_DELAY = 0
    return f


def test_pagination_merges_race_split_across_pages(fetcher, monkeypatch):
    pages = {
        0: {"MRData": {"total": "5", "RaceTable": {"Races": [_race_payload(2025, 1, [_row("a", 1), _row("b", 2)]), _race_payload(2025, 2, [_row("a", 1)])]}}},
        3: {"MRData": {"total": "5", "RaceTable": {"Races": [_race_payload(2025, 2, [_row("b", 2)]), _race_payload(2025, 3, [_row("c", 1)])]}}},
    }
    calls = []

    def fake_request(url, params=None):
        calls.append(params["offset"])
        return pages[params["offset"]]

    fetcher.PAGE_SIZE = 3
    monkeypatch.setattr(fetcher, "_make_request", fake_request)
    results = fetcher.get_season_results(2025)
    assert calls == [0, 3]
    by_round = {}
    for r in results:
        by_round.setdefault(r.race.round, []).append(r.driver.driver_id)
    assert by_round == {1: ["a", "b"], 2: ["a", "b"], 3: ["c"]}
    assert results[0].race.circuit.latitude == 1.5

    # Second call is served from the cache
    calls.clear()
    fetcher.get_season_results(2025)
    assert calls == []


def test_parse_race_flags_sprint_and_qualifying(fetcher, monkeypatch):
    payload = {"MRData": {"total": "2", "RaceTable": {"Races": [
        _race_payload(2025, 2, [{"Driver": _row("a", 1)["Driver"], "Constructor": _row("a", 1)["Constructor"], "position": "1", "Q1": "1:20.0"}], key="QualifyingResults", sprint=True)
    ]}}}
    monkeypatch.setattr(fetcher, "_make_request", lambda url, params=None: payload)
    quali = fetcher.get_season_qualifying(2025)
    assert len(quali) == 1 and quali[0].position == 1 and quali[0].q1_time == "1:20.0"
    assert quali[0].race.is_sprint
    assert fetcher.get_qualifying_results(2025, 2) == quali
    assert fetcher.get_qualifying_results(2025, 3) == []


def test_schedule_and_standings(fetcher, monkeypatch):
    def fake_request(url, params=None):
        if url.endswith("/2025.json"):
            return {"MRData": {"RaceTable": {"Races": [_race_payload(2025, 1, []), _race_payload(2025, 2, [], sprint=True)]}}}
        if "driverStandings" in url:
            return {"MRData": {"StandingsTable": {"StandingsLists": [{"round": "2", "DriverStandings": [
                {"Driver": _row("a", 1)["Driver"], "Constructors": [_row("a", 1)["Constructor"]], "position": "1", "points": "43", "wins": "2"}]}]}}}
        raise AssertionError(url)

    monkeypatch.setattr(fetcher, "_make_request", fake_request)
    schedule = fetcher.get_season_schedule(2025)
    assert [r.round for r in schedule] == [1, 2]
    assert schedule[1].is_sprint and not schedule[0].is_sprint
    assert schedule[0].total_rounds == 2
    standings = fetcher.get_driver_standings(2025, 2)
    assert standings[0].points == 43 and standings[0].wins == 2


def test_stale_cache_used_on_network_failure(fetcher, monkeypatch):
    payload = {"MRData": {"total": "1", "RaceTable": {"Races": [_race_payload(2025, 1, [_row("a", 1)])]}}}
    monkeypatch.setattr(fetcher, "_make_request", lambda url, params=None: payload)
    assert len(fetcher.get_season_results(2025)) == 1
    # expire the cache entry and make the network fail
    fetcher.cache.set("season_results_2025", fetcher.cache.get("season_results_2025"), ttl=-1)

    def boom(url, params=None):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(fetcher, "_make_request", boom)
    assert len(fetcher.get_season_results(2025)) == 1


def test_driver_without_code_gets_fallback(fetcher):
    d = fetcher._parse_driver({"driverId": "x", "givenName": "Jean", "familyName": "Alesi"})
    assert d.code == "ALE"
