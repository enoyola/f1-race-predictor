"""
F1 data fetcher module.

Retrieves data from the Jolpica F1 API (an Ergast-compatible replacement) with
pagination, rate limiting, retries, and caching.

Jolpica caps every response at 100 rows, so season-wide endpoints are fetched
page by page and merged. A single race's rows can straddle a page boundary, so
pages are merged by round rather than concatenated.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from f1_predictor.cache import DataCache
from f1_predictor.models import (
    Circuit, Constructor, ConstructorStanding, Driver, DriverStanding,
    QualifyingResult, Race, RaceResult
)

logger = logging.getLogger(__name__)


class F1DataFetcher:
    """Fetches F1 data from the Jolpica F1 API."""

    BASE_URL = "https://api.jolpi.ca/ergast/f1"
    REQUEST_TIMEOUT = 15          # seconds
    RATE_LIMIT_DELAY = 0.3        # Jolpica allows 4 requests/second unauthenticated
    MAX_RETRIES = 4
    PAGE_SIZE = 100               # hard cap imposed by the API

    # Cache TTLs in seconds
    TTL_CURRENT_SEASON_RESULTS = 6 * 3600
    TTL_CURRENT_SEASON_QUALIFYING = 3600
    TTL_CURRENT_SEASON_SCHEDULE = 24 * 3600
    TTL_COMPLETED_SEASON = 30 * 24 * 3600
    TTL_NEXT_RACE = 3600
    TTL_STANDINGS = 6 * 3600

    def __init__(self, cache: Optional[DataCache] = None, use_cache: bool = True):
        self.cache = cache if cache else DataCache()
        self.use_cache = use_cache
        self._last_request_time = 0.0
        self.request_count = 0

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an HTTP GET with retries, returning the parsed MRData payload."""
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            self._rate_limit()
            try:
                self.request_count += 1
                response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else 5.0 * attempt
                    logger.warning(f"Rate limited by API, waiting {wait:.0f}s (attempt {attempt}/{self.MAX_RETRIES})")
                    time.sleep(wait)
                    last_exception = requests.HTTPError("429 Too Many Requests", response=response)
                    continue

                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or 'MRData' not in data:
                    raise requests.RequestException("API response missing 'MRData'")
                return data

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status is not None and 400 <= status < 500 and status != 429:
                    logger.error(f"Client error {status}: {url}")
                    raise
                last_exception = e
                logger.warning(f"Server error (attempt {attempt}/{self.MAX_RETRIES}): {url} - {e}")
            except (requests.RequestException, ValueError) as e:
                last_exception = e
                logger.warning(f"Request failed (attempt {attempt}/{self.MAX_RETRIES}): {url} - {e}")

            if attempt < self.MAX_RETRIES:
                time.sleep(min(2 ** attempt, 20))

        logger.error(f"Request failed after {self.MAX_RETRIES} attempts: {url}")
        if isinstance(last_exception, Exception):
            raise last_exception
        raise requests.RequestException(f"Request failed: {url}")

    def _get_cached_or_fetch(
        self,
        cache_key: str,
        fetch_fn,
        ttl: int,
    ) -> Any:
        """Return cached data for the key, or call fetch_fn() and cache its result."""
        if self.use_cache:
            try:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached
            except Exception as e:
                logger.warning(f"Cache read error for {cache_key}: {e}")

        try:
            data = fetch_fn()
        except requests.RequestException:
            if self.use_cache:
                stale = self.cache.get(cache_key, ignore_ttl=True)
                if stale is not None:
                    logger.warning(f"API request failed, using stale cache for {cache_key}")
                    return stale
            raise

        if self.use_cache:
            try:
                self.cache.set(cache_key, data, ttl)
            except Exception as e:
                logger.warning(f"Cache write error for {cache_key}: {e}")
        return data

    def _fetch_races_paginated(self, url: str, list_key: str) -> List[Dict[str, Any]]:
        """
        Fetch every page of a RaceTable endpoint and merge rows by round.

        Args:
            url: Endpoint URL (without pagination params)
            list_key: Name of the per-race list to merge ("Results", "QualifyingResults", "SprintResults")
        """
        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        offset = 0
        total: Optional[int] = None

        while True:
            data = self._make_request(url, params={"limit": self.PAGE_SIZE, "offset": offset})
            mr = data['MRData']
            races = mr.get('RaceTable', {}).get('Races', [])
            total = int(mr.get('total', 0))
            fetched_rows = 0

            for race in races:
                key = f"{race.get('season')}_{race.get('round')}"
                rows = race.get(list_key, [])
                fetched_rows += len(rows)
                if key in merged:
                    merged[key][list_key].extend(rows)
                else:
                    race_copy = dict(race)
                    race_copy[list_key] = list(rows)
                    merged[key] = race_copy
                    order.append(key)

            offset += self.PAGE_SIZE
            if fetched_rows == 0 or offset >= total:
                break

        return [merged[k] for k in order]

    # ------------------------------------------------------------------
    # TTL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _season_is_complete(season: int) -> bool:
        return season < datetime.now().year

    def _ttl_for(self, season: int, current_ttl: int) -> int:
        return self.TTL_COMPLETED_SEASON if self._season_is_complete(season) else current_ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_next_race(self) -> Race:
        """Get the next scheduled F1 race."""
        url = f"{self.BASE_URL}/current/next.json"
        try:
            data = self._get_cached_or_fetch("next_race", lambda: self._make_request(url), self.TTL_NEXT_RACE)
        except requests.RequestException as e:
            raise ValueError(f"Unable to fetch next race information: {e}")

        races = data['MRData'].get('RaceTable', {}).get('Races', [])
        if not races:
            raise ValueError("No upcoming races found in API response")
        race = self._parse_race(races[0])
        try:
            schedule = self.get_season_schedule(race.season)
            race.total_rounds = len(schedule)
            for scheduled in schedule:
                if scheduled.round == race.round:
                    race.is_sprint = scheduled.is_sprint
        except Exception as e:  # schedule is optional decoration
            logger.debug(f"Could not decorate next race with schedule info: {e}")
        return race

    def get_current_season(self) -> int:
        """Return the season the API considers current."""
        try:
            return self.get_next_race().season
        except ValueError:
            return datetime.now().year

    def get_season_schedule(self, season: int) -> List[Race]:
        """Get every scheduled race of a season, including future rounds."""
        url = f"{self.BASE_URL}/{season}.json"
        data = self._get_cached_or_fetch(
            f"schedule_{season}",
            lambda: self._make_request(url, params={"limit": self.PAGE_SIZE}),
            self._ttl_for(season, self.TTL_CURRENT_SEASON_SCHEDULE),
        )
        races_data = data['MRData'].get('RaceTable', {}).get('Races', [])
        races: List[Race] = []
        for race_data in races_data:
            try:
                races.append(self._parse_race(race_data))
            except ValueError as e:
                logger.warning(f"Skipping malformed race in schedule: {e}")
        for race in races:
            race.total_rounds = len(races)
        return races

    def get_season_results(self, season: int) -> List[RaceResult]:
        """Get every race result of a season (all rounds run so far)."""
        url = f"{self.BASE_URL}/{season}/results.json"
        races_data = self._get_cached_or_fetch(
            f"season_results_{season}",
            lambda: self._fetch_races_paginated(url, "Results"),
            self._ttl_for(season, self.TTL_CURRENT_SEASON_RESULTS),
        )
        return self._parse_result_rows(races_data, "Results")

    # Backward-compatible alias
    def get_current_season_results(self, season: Optional[int] = None) -> List[RaceResult]:
        return self.get_season_results(season or self.get_current_season())

    def get_season_sprint_results(self, season: int) -> List[RaceResult]:
        """Get every sprint result of a season."""
        url = f"{self.BASE_URL}/{season}/sprint.json"
        try:
            races_data = self._get_cached_or_fetch(
                f"season_sprints_{season}",
                lambda: self._fetch_races_paginated(url, "SprintResults"),
                self._ttl_for(season, self.TTL_CURRENT_SEASON_QUALIFYING),
            )
        except requests.HTTPError as e:
            # Seasons before 2021 have no sprint endpoint data
            logger.debug(f"No sprint data for {season}: {e}")
            return []
        return self._parse_result_rows(races_data, "SprintResults")

    def get_season_qualifying(self, season: int) -> List[QualifyingResult]:
        """Get every qualifying result of a season."""
        url = f"{self.BASE_URL}/{season}/qualifying.json"
        races_data = self._get_cached_or_fetch(
            f"season_qualifying_{season}",
            lambda: self._fetch_races_paginated(url, "QualifyingResults"),
            self._ttl_for(season, self.TTL_CURRENT_SEASON_QUALIFYING),
        )
        results: List[QualifyingResult] = []
        for race_data in races_data:
            try:
                race = self._parse_race(race_data)
            except ValueError as e:
                logger.warning(f"Skipping malformed qualifying race: {e}")
                continue
            for row in race_data.get("QualifyingResults", []):
                try:
                    results.append(self._parse_qualifying_result(race, row))
                except ValueError as e:
                    logger.warning(f"Skipping malformed qualifying row: {e}")
        return results

    def get_qualifying_results(self, season: int, round_num: int) -> List[QualifyingResult]:
        """Get qualifying results for one race (empty if not yet available)."""
        return [q for q in self.get_season_qualifying(season) if q.race.round == round_num]

    def get_driver_standings(self, season: Optional[int] = None, round_num: Optional[int] = None) -> List[DriverStanding]:
        """Get official driver standings for a season, optionally after a given round."""
        season_str = str(season) if season else "current"
        path = f"{season_str}/{round_num}" if round_num else season_str
        url = f"{self.BASE_URL}/{path}/driverStandings.json"
        try:
            data = self._get_cached_or_fetch(
                f"driver_standings_{path.replace('/', '_')}",
                lambda: self._make_request(url),
                self.TTL_STANDINGS if not (season and round_num) else self.TTL_COMPLETED_SEASON,
            )
        except requests.RequestException as e:
            logger.error(f"Failed to fetch driver standings: {e}")
            return []
        lists = data['MRData'].get('StandingsTable', {}).get('StandingsLists', [])
        if not lists:
            return []
        standings = []
        for row in lists[0].get('DriverStandings', []):
            try:
                standings.append(self._parse_driver_standing(row))
            except ValueError as e:
                logger.warning(f"Skipping malformed driver standing: {e}")
        return standings

    def get_constructor_standings(self, season: Optional[int] = None, round_num: Optional[int] = None) -> List[ConstructorStanding]:
        """Get official constructor standings for a season, optionally after a given round."""
        season_str = str(season) if season else "current"
        path = f"{season_str}/{round_num}" if round_num else season_str
        url = f"{self.BASE_URL}/{path}/constructorStandings.json"
        try:
            data = self._get_cached_or_fetch(
                f"constructor_standings_{path.replace('/', '_')}",
                lambda: self._make_request(url),
                self.TTL_STANDINGS if not (season and round_num) else self.TTL_COMPLETED_SEASON,
            )
        except requests.RequestException as e:
            logger.error(f"Failed to fetch constructor standings: {e}")
            return []
        lists = data['MRData'].get('StandingsTable', {}).get('StandingsLists', [])
        if not lists:
            return []
        standings = []
        for row in lists[0].get('ConstructorStandings', []):
            try:
                standings.append(self._parse_constructor_standing(row))
            except ValueError as e:
                logger.warning(f"Skipping malformed constructor standing: {e}")
        return standings

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_result_rows(self, races_data: List[Dict[str, Any]], list_key: str) -> List[RaceResult]:
        results: List[RaceResult] = []
        for race_data in races_data:
            try:
                race = self._parse_race(race_data)
            except ValueError as e:
                logger.warning(f"Skipping malformed race: {e}")
                continue
            for row in race_data.get(list_key, []):
                try:
                    results.append(self._parse_race_result(race, row))
                except ValueError as e:
                    logger.warning(f"Skipping malformed result row: {e}")
        return results

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _parse_circuit(self, circuit_data: Dict[str, Any]) -> Circuit:
        for field in ('circuitId', 'circuitName', 'Location'):
            if field not in circuit_data:
                raise ValueError(f"Circuit data missing required field: {field}")
        location = circuit_data['Location']
        return Circuit(
            circuit_id=circuit_data['circuitId'],
            circuit_name=circuit_data['circuitName'],
            location=location.get('locality', ''),
            country=location.get('country', ''),
            latitude=self._optional_float(location.get('lat')),
            longitude=self._optional_float(location.get('long')),
        )

    def _parse_driver(self, driver_data: Dict[str, Any]) -> Driver:
        for field in ('driverId', 'givenName', 'familyName'):
            if field not in driver_data:
                raise ValueError(f"Driver data missing required field: {field}")
        return Driver(
            driver_id=driver_data['driverId'],
            code=driver_data.get('code') or driver_data['familyName'][:3].upper(),
            forename=driver_data['givenName'],
            surname=driver_data['familyName'],
            nationality=driver_data.get('nationality', ''),
        )

    def _parse_constructor(self, constructor_data: Dict[str, Any]) -> Constructor:
        for field in ('constructorId', 'name'):
            if field not in constructor_data:
                raise ValueError(f"Constructor data missing required field: {field}")
        return Constructor(
            constructor_id=constructor_data['constructorId'],
            name=constructor_data['name'],
            nationality=constructor_data.get('nationality', ''),
        )

    def _parse_race(self, race_data: Dict[str, Any]) -> Race:
        for field in ('season', 'round', 'raceName', 'Circuit', 'date'):
            if field not in race_data:
                raise ValueError(f"Race data missing required field: {field}")
        try:
            time_str = race_data.get('time') or '00:00:00Z'
            race_date = datetime.fromisoformat(f"{race_data['date']}T{time_str.rstrip('Z')}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid race date/time format: {e}")
        return Race(
            season=int(race_data['season']),
            round=int(race_data['round']),
            race_name=race_data['raceName'],
            circuit=self._parse_circuit(race_data['Circuit']),
            date=race_date,
            is_sprint='Sprint' in race_data or 'SprintResults' in race_data,
        )

    def _parse_race_result(self, race: Race, result_data: Dict[str, Any]) -> RaceResult:
        for field in ('Driver', 'Constructor', 'position'):
            if field not in result_data:
                raise ValueError(f"Race result data missing required field: {field}")
        try:
            return RaceResult(
                race=race,
                driver=self._parse_driver(result_data['Driver']),
                constructor=self._parse_constructor(result_data['Constructor']),
                position=int(result_data['position']),
                points=float(result_data.get('points', 0) or 0),
                grid=int(result_data.get('grid', 0) or 0),
                laps=int(result_data.get('laps', 0) or 0),
                status=result_data.get('status', ''),
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid race result data types: {e}")

    def _parse_qualifying_result(self, race: Race, qual_data: Dict[str, Any]) -> QualifyingResult:
        for field in ('Driver', 'Constructor', 'position'):
            if field not in qual_data:
                raise ValueError(f"Qualifying result data missing required field: {field}")
        try:
            return QualifyingResult(
                race=race,
                driver=self._parse_driver(qual_data['Driver']),
                constructor=self._parse_constructor(qual_data['Constructor']),
                position=int(qual_data['position']),
                q1_time=qual_data.get('Q1'),
                q2_time=qual_data.get('Q2'),
                q3_time=qual_data.get('Q3'),
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid qualifying result data types: {e}")

    def _parse_driver_standing(self, standing_data: Dict[str, Any]) -> DriverStanding:
        for field in ('Driver', 'Constructors', 'position', 'points', 'wins'):
            if field not in standing_data:
                raise ValueError(f"Driver standing data missing required field: {field}")
        if not standing_data['Constructors']:
            raise ValueError("Driver standing missing constructor information")
        try:
            return DriverStanding(
                driver=self._parse_driver(standing_data['Driver']),
                constructor=self._parse_constructor(standing_data['Constructors'][-1]),
                position=int(standing_data['position']),
                points=float(standing_data['points']),
                wins=int(standing_data['wins']),
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid driver standing data types: {e}")

    def _parse_constructor_standing(self, standing_data: Dict[str, Any]) -> ConstructorStanding:
        for field in ('Constructor', 'position', 'points', 'wins'):
            if field not in standing_data:
                raise ValueError(f"Constructor standing data missing required field: {field}")
        try:
            return ConstructorStanding(
                constructor=self._parse_constructor(standing_data['Constructor']),
                position=int(standing_data['position']),
                points=float(standing_data['points']),
                wins=int(standing_data['wins']),
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid constructor standing data types: {e}")
