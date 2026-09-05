"""
Point-in-time race contexts.

The HistoricalDataStore loads whole seasons from the API (cached) and builds a
RaceContext for any race that only contains information available before that
race started. The same builder serves live predictions, model training and
backtests.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from f1_predictor.data_fetcher import F1DataFetcher
from f1_predictor.models import (
    Constructor, ConstructorStanding, Driver, DriverStanding, Entry,
    QualifyingResult, Race, RaceContext, RaceResult
)
from f1_predictor.weather import WeatherClient

logger = logging.getLogger(__name__)


@dataclass
class SeasonData:
    season: int
    schedule: List[Race]
    results_by_round: Dict[int, List[RaceResult]] = field(default_factory=dict)
    sprints_by_round: Dict[int, List[RaceResult]] = field(default_factory=dict)
    qualifying_by_round: Dict[int, List[QualifyingResult]] = field(default_factory=dict)
    qualifying_loaded: bool = False

    @property
    def schedule_by_round(self) -> Dict[int, Race]:
        return {r.round: r for r in self.schedule}

    @property
    def completed_rounds(self) -> List[int]:
        return sorted(self.results_by_round.keys())

    @property
    def total_rounds(self) -> int:
        return len(self.schedule)


def _sort_key_driver(points: float, wins: int, podiums: int) -> tuple:
    return (-points, -wins, -podiums)


def compute_driver_standings(
    results_by_round: Dict[int, List[RaceResult]],
    sprints_by_round: Dict[int, List[RaceResult]],
    before_round: Optional[int] = None,
) -> List[DriverStanding]:
    """Driver standings computed from race and sprint points before a round."""
    points: Dict[str, float] = {}
    wins: Dict[str, int] = {}
    podiums: Dict[str, int] = {}
    drivers: Dict[str, Driver] = {}
    constructor_of: Dict[str, Constructor] = {}
    latest_round: Dict[str, int] = {}

    def include(rnd: int) -> bool:
        return before_round is None or rnd < before_round

    for rnd, results in results_by_round.items():
        if not include(rnd):
            continue
        for r in results:
            d = r.driver.driver_id
            drivers[d] = r.driver
            points[d] = points.get(d, 0.0) + r.points
            if r.position == 1:
                wins[d] = wins.get(d, 0) + 1
            if r.position <= 3:
                podiums[d] = podiums.get(d, 0) + 1
            if rnd >= latest_round.get(d, -1):
                latest_round[d] = rnd
                constructor_of[d] = r.constructor
    for rnd, results in sprints_by_round.items():
        if not include(rnd):
            continue
        for r in results:
            d = r.driver.driver_id
            drivers.setdefault(d, r.driver)
            points[d] = points.get(d, 0.0) + r.points
            if d not in constructor_of:
                constructor_of[d] = r.constructor

    ordered = sorted(
        drivers.keys(),
        key=lambda d: _sort_key_driver(points.get(d, 0.0), wins.get(d, 0), podiums.get(d, 0)) + (drivers[d].surname,),
    )
    return [
        DriverStanding(
            driver=drivers[d],
            constructor=constructor_of[d],
            position=i + 1,
            points=points.get(d, 0.0),
            wins=wins.get(d, 0),
        )
        for i, d in enumerate(ordered)
    ]


def compute_constructor_standings(
    results_by_round: Dict[int, List[RaceResult]],
    sprints_by_round: Dict[int, List[RaceResult]],
    before_round: Optional[int] = None,
) -> List[ConstructorStanding]:
    """Constructor standings computed from race and sprint points before a round."""
    points: Dict[str, float] = {}
    wins: Dict[str, int] = {}
    podiums: Dict[str, int] = {}
    constructors: Dict[str, Constructor] = {}

    def include(rnd: int) -> bool:
        return before_round is None or rnd < before_round

    for rnd, results in results_by_round.items():
        if not include(rnd):
            continue
        for r in results:
            c = r.constructor.constructor_id
            constructors[c] = r.constructor
            points[c] = points.get(c, 0.0) + r.points
            if r.position == 1:
                wins[c] = wins.get(c, 0) + 1
            if r.position <= 3:
                podiums[c] = podiums.get(c, 0) + 1
    for rnd, results in sprints_by_round.items():
        if not include(rnd):
            continue
        for r in results:
            c = r.constructor.constructor_id
            constructors.setdefault(c, r.constructor)
            points[c] = points.get(c, 0.0) + r.points

    ordered = sorted(
        constructors.keys(),
        key=lambda c: _sort_key_driver(points.get(c, 0.0), wins.get(c, 0), podiums.get(c, 0)) + (constructors[c].name,),
    )
    return [
        ConstructorStanding(
            constructor=constructors[c],
            position=i + 1,
            points=points.get(c, 0.0),
            wins=wins.get(c, 0),
        )
        for i, c in enumerate(ordered)
    ]


class HistoricalDataStore:
    """Loads seasons on demand and builds point-in-time race contexts."""

    def __init__(
        self,
        fetcher: F1DataFetcher,
        weather_client: Optional[WeatherClient] = None,
        history_years: int = 5,
        min_season: int = 1950,
    ):
        self.fetcher = fetcher
        self.weather_client = weather_client
        self.history_years = history_years
        self.min_season = min_season
        self._seasons: Dict[int, SeasonData] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_season(self, season: int, with_qualifying: bool = False) -> SeasonData:
        """Load schedule, results and sprints for a season (qualifying on request)."""
        data = self._seasons.get(season)
        if data is None:
            logger.info(f"Loading season {season}")
            schedule = self.fetcher.get_season_schedule(season)
            data = SeasonData(season=season, schedule=schedule)
            for r in self.fetcher.get_season_results(season):
                data.results_by_round.setdefault(r.race.round, []).append(r)
            for s in self.fetcher.get_season_sprint_results(season):
                data.sprints_by_round.setdefault(s.race.round, []).append(s)
            for rnd, rows in data.results_by_round.items():
                rows.sort(key=lambda r: r.position)
                # Decorate result races with schedule metadata
                scheduled = data.schedule_by_round.get(rnd)
                for row in rows:
                    row.race.total_rounds = data.total_rounds
                    if scheduled is not None:
                        row.race.is_sprint = scheduled.is_sprint
            self._seasons[season] = data
        if with_qualifying and not data.qualifying_loaded:
            for q in self.fetcher.get_season_qualifying(season):
                data.qualifying_by_round.setdefault(q.race.round, []).append(q)
            for rows in data.qualifying_by_round.values():
                rows.sort(key=lambda q: q.position)
            data.qualifying_loaded = True
        return data

    def invalidate(self, season: int) -> None:
        self._seasons.pop(season, None)

    def completed_rounds(self, season: int) -> List[int]:
        return self.load_season(season).completed_rounds

    def schedule(self, season: int) -> List[Race]:
        return self.load_season(season).schedule

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _previous_season(self, season: int) -> Optional[SeasonData]:
        prev = season - 1
        if prev < self.min_season:
            return None
        try:
            data = self.load_season(prev)
        except Exception as e:
            logger.warning(f"Could not load previous season {prev}: {e}")
            return None
        return data if data.results_by_round else None

    def _entries_from_results(self, results: List[RaceResult]) -> List[Entry]:
        seen = set()
        entries = []
        for r in results:
            if r.driver.driver_id in seen:
                continue
            seen.add(r.driver.driver_id)
            entries.append(Entry(driver=r.driver, constructor=r.constructor))
        return entries

    def _circuit_history(self, race: Race, exclude_round: Optional[int] = None) -> List[RaceResult]:
        history: List[RaceResult] = []
        for season in range(race.season - self.history_years, race.season + 1):
            if season < self.min_season:
                continue
            try:
                data = self.load_season(season)
            except Exception as e:
                logger.warning(f"Could not load season {season} for circuit history: {e}")
                continue
            for rnd, results in data.results_by_round.items():
                if not results:
                    continue
                past_race = results[0].race
                if past_race.circuit.circuit_id != race.circuit.circuit_id:
                    continue
                if past_race.date >= race.date:
                    continue
                history.extend(results)
        return history

    def build_context(
        self,
        season: int,
        round_num: int,
        include_actual: bool = False,
        use_weather: bool = True,
    ) -> RaceContext:
        """
        Build the pre-race context for a race.

        Args:
            season: Season year
            round_num: Round number within the season
            include_actual: Attach the actual results (for backtests and reviews)
            use_weather: Look up race-day weather
        """
        data = self.load_season(season, with_qualifying=True)
        race = data.schedule_by_round.get(round_num)
        if race is None:
            raise ValueError(f"Round {round_num} not found in the {season} schedule")
        race.total_rounds = data.total_rounds

        prior_rounds = [r for r in data.completed_rounds if r < round_num]
        notes: List[str] = []

        # Standings before this race
        from_previous = False
        if prior_rounds:
            driver_standings = compute_driver_standings(data.results_by_round, data.sprints_by_round, before_round=round_num)
            constructor_standings = compute_constructor_standings(data.results_by_round, data.sprints_by_round, before_round=round_num)
        else:
            prev = self._previous_season(season)
            if prev is not None:
                driver_standings = compute_driver_standings(prev.results_by_round, prev.sprints_by_round)
                constructor_standings = compute_constructor_standings(prev.results_by_round, prev.sprints_by_round)
                from_previous = True
                notes.append(f"Season opener: standings taken from the end of {prev.season}")
            else:
                driver_standings, constructor_standings = [], []
                notes.append("No standings available")

        # Recent results: previous season plus this season's earlier rounds
        recent: List[RaceResult] = []
        prev = self._previous_season(season)
        if prev is not None:
            for results in prev.results_by_round.values():
                recent.extend(results)
        for rnd in prior_rounds:
            recent.extend(data.results_by_round[rnd])
        recent.sort(key=lambda r: r.race.date)

        qualifying = data.qualifying_by_round.get(round_num, [])
        sprint = data.sprints_by_round.get(round_num, [])
        actual = data.results_by_round.get(round_num) if include_actual else None

        # Entry list
        if qualifying:
            entries = self._entries_from_results(
                [RaceResult(q.race, q.driver, q.constructor, q.position, 0.0, q.position, 0, "") for q in qualifying]
            )
        elif sprint:
            entries = self._entries_from_results(sprint)
        elif prior_rounds:
            entries = self._entries_from_results(data.results_by_round[prior_rounds[-1]])
            notes.append("Entry list taken from the previous round")
        elif prev is not None and prev.completed_rounds:
            entries = self._entries_from_results(prev.results_by_round[prev.completed_rounds[-1]])
            notes.append(f"Entry list taken from the final round of {prev.season}")
        else:
            entries = []
        if not entries and actual:
            entries = self._entries_from_results(actual)

        # Update entries' constructors with the latest known pairing when the
        # entry list came from an older race than the qualifying session.
        if not qualifying and prior_rounds:
            latest_constructor = {r.driver.driver_id: r.constructor for r in data.results_by_round[prior_rounds[-1]]}
            for e in entries:
                if e.driver.driver_id in latest_constructor:
                    e.constructor = latest_constructor[e.driver.driver_id]

        weather = None
        if use_weather and self.weather_client is not None:
            try:
                weather = self.weather_client.get_race_weather(race)
            except Exception as e:
                logger.debug(f"Weather lookup failed: {e}")

        if not qualifying:
            notes.append("Qualifying not yet available")

        return RaceContext(
            race=race,
            entries=entries,
            driver_standings=driver_standings,
            constructor_standings=constructor_standings,
            recent_results=recent,
            qualifying_results=qualifying,
            sprint_results=sprint,
            circuit_history=self._circuit_history(race),
            weather=weather,
            rounds_completed=len(prior_rounds),
            standings_from_previous_season=from_previous,
            actual_results=actual,
            notes=notes,
        )
