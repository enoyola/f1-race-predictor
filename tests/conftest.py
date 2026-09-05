"""Shared fixtures: a small synthetic F1 world served by an in-memory fetcher."""

from datetime import datetime, timedelta
from typing import Dict, List

import pytest

from f1_predictor.history import HistoricalDataStore
from f1_predictor.models import (
    Circuit, Constructor, Driver, Entry, QualifyingResult, Race, RaceContext, RaceResult,
    Weather,
)

POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

TEAMS = [
    Constructor("alpha", "Alpha Racing", "British"),
    Constructor("beta", "Beta GP", "Italian"),
    Constructor("gamma", "Gamma F1", "German"),
]
DRIVERS = [
    (Driver("aaa", "AAA", "Ann", "Alpha-One", "British"), TEAMS[0]),
    (Driver("aab", "AAB", "Al", "Alpha-Two", "Dutch"), TEAMS[0]),
    (Driver("bba", "BBA", "Bea", "Beta-One", "Italian"), TEAMS[1]),
    (Driver("bbb", "BBB", "Bo", "Beta-Two", "French"), TEAMS[1]),
    (Driver("gga", "GGA", "Gia", "Gamma-One", "German"), TEAMS[2]),
    (Driver("ggb", "GGB", "Gus", "Gamma-Two", "Spanish"), TEAMS[2]),
]
CIRCUITS = [
    Circuit("c_one", "Circuit One", "Townsville", "Utopia", 10.0, 20.0),
    Circuit("c_two", "Circuit Two", "Lakeside", "Utopia", 11.0, 21.0),
    Circuit("c_three", "Circuit Three", "Hillcrest", "Arcadia", 12.0, 22.0),
]


def make_race(season: int, rnd: int, circuit: Circuit = None, is_sprint: bool = False, total_rounds: int = 6) -> Race:
    return Race(
        season=season,
        round=rnd,
        race_name=f"{(circuit or CIRCUITS[(rnd - 1) % 3]).country} Grand Prix {rnd}",
        circuit=circuit or CIRCUITS[(rnd - 1) % 3],
        date=datetime(season, 3, 1) + timedelta(days=14 * (rnd - 1)),
        is_sprint=is_sprint,
        total_rounds=total_rounds,
    )


def make_results(race: Race, order: List[int], statuses: Dict[int, str] = None, grid: List[int] = None) -> List[RaceResult]:
    """order: list of driver indexes in finishing order."""
    statuses = statuses or {}
    results = []
    for pos, idx in enumerate(order, 1):
        driver, team = DRIVERS[idx]
        status = statuses.get(idx, "Finished")
        results.append(RaceResult(
            race=race, driver=driver, constructor=team, position=pos,
            points=float(POINTS.get(pos, 0)) if status in ("Finished", "Lapped") else 0.0,
            grid=(grid[idx] if grid else pos), laps=50, status=status,
        ))
    return results


def make_qualifying(race: Race, order: List[int]) -> List[QualifyingResult]:
    return [
        QualifyingResult(race=race, driver=DRIVERS[idx][0], constructor=DRIVERS[idx][1], position=pos,
                         q1_time="1:30.0", q2_time="1:29.0", q3_time="1:28.0")
        for pos, idx in enumerate(order, 1)
    ]


def make_sprint(race: Race, order: List[int]) -> List[RaceResult]:
    return [
        RaceResult(race=race, driver=DRIVERS[idx][0], constructor=DRIVERS[idx][1], position=pos,
                   points=float(SPRINT_POINTS.get(pos, 0)), grid=pos, laps=20, status="Finished")
        for pos, idx in enumerate(order, 1)
    ]


class FakeFetcher:
    """In-memory stand-in for F1DataFetcher."""

    def __init__(self):
        self.schedules: Dict[int, List[Race]] = {}
        self.results: Dict[int, List[RaceResult]] = {}
        self.sprints: Dict[int, List[RaceResult]] = {}
        self.qualifying: Dict[int, List[QualifyingResult]] = {}
        self.calls: List[str] = []

    def add_season(self, season: int, rounds: int, completed: int, sprint_rounds=(), winners=None):
        """
        winners: mapping round -> finishing order (driver indexes). Defaults to a
        fixed order rotated so different drivers win.
        """
        schedule = [make_race(season, r, is_sprint=(r in sprint_rounds), total_rounds=rounds) for r in range(1, rounds + 1)]
        self.schedules[season] = schedule
        self.results.setdefault(season, [])
        self.sprints.setdefault(season, [])
        self.qualifying.setdefault(season, [])
        for race in schedule:
            if race.round > completed + 1:
                continue
            order = (winners or {}).get(race.round) or [0, 2, 1, 4, 3, 5]
            self.qualifying[season].extend(make_qualifying(race, order))
            if race.is_sprint and race.round <= completed + 1:
                self.sprints[season].extend(make_sprint(race, order))
            if race.round <= completed:
                self.results[season].extend(make_results(race, order))

    def get_season_schedule(self, season):
        self.calls.append(f"schedule_{season}")
        return list(self.schedules.get(season, []))

    def get_season_results(self, season):
        self.calls.append(f"results_{season}")
        return list(self.results.get(season, []))

    def get_season_sprint_results(self, season):
        self.calls.append(f"sprints_{season}")
        return list(self.sprints.get(season, []))

    def get_season_qualifying(self, season):
        self.calls.append(f"qualifying_{season}")
        return list(self.qualifying.get(season, []))

    def get_current_season(self):
        return max(self.schedules)


@pytest.fixture
def fake_fetcher() -> FakeFetcher:
    f = FakeFetcher()
    # Previous season fully complete; driver 2 (bba) wins rounds 5 and 6
    f.add_season(2024, rounds=6, completed=6, winners={5: [2, 0, 1, 3, 4, 5], 6: [2, 1, 0, 3, 4, 5]})
    # Current season: 4 of 6 rounds done, round 3 is a sprint weekend, round 5 has qualifying only
    f.add_season(2025, rounds=6, completed=4, sprint_rounds=(3,), winners={2: [1, 0, 2, 3, 4, 5]})
    return f


@pytest.fixture
def store(fake_fetcher) -> HistoricalDataStore:
    return HistoricalDataStore(fake_fetcher, weather_client=None, history_years=3, min_season=2024)


@pytest.fixture
def sample_context(store) -> RaceContext:
    """2025 round 5: four rounds done, qualifying available, no sprint."""
    return store.build_context(2025, 5)


@pytest.fixture
def wet_context(sample_context) -> RaceContext:
    from dataclasses import replace
    return replace(sample_context, weather=Weather(date="2025-04-26", precipitation_mm=5.0, precipitation_probability=None,
                                                   temperature_max=18.0, source="archive"))
