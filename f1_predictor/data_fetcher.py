"""
F1 data fetcher module.

Retrieves data from the Jolpica F1 API with error handling, retry logic,
and caching support.

Jolpica F1 API is a modern replacement for the discontinued Ergast API,
maintaining the same API structure and endpoints for backward compatibility.
"""

import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import requests

from f1_predictor.models import (
    Race, Circuit, Driver, Constructor, RaceResult,
    QualifyingResult, DriverStanding, ConstructorStanding
)
from f1_predictor.cache import DataCache


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1DataFetcher:
    """
    Fetches F1 data from the Jolpica F1 API.
    
    Includes error handling, retry logic, rate limiting, and caching support.
    Jolpica maintains Ergast API compatibility while providing updated data.
    """
    
    BASE_URL = "https://api.jolpi.ca/ergast/f1"
    REQUEST_TIMEOUT = 10  # seconds
    RATE_LIMIT_DELAY = 0.1  # 100ms between requests
    MAX_RETRIES = 3
    
    # Cache TTL values (in seconds)
    CACHE_TTL_SEASON_RESULTS = 86400  # 24 hours
    CACHE_TTL_STANDINGS = 21600  # 6 hours
    CACHE_TTL_QUALIFYING = 3600  # 1 hour
    CACHE_TTL_CIRCUIT_HISTORY = 604800  # 7 days
    CACHE_TTL_NEXT_RACE = 3600  # 1 hour
    
    def __init__(self, cache: Optional[DataCache] = None, use_cache: bool = True):
        """
        Initialize F1 data fetcher.
        
        Args:
            cache: DataCache instance for caching (creates default if None)
            use_cache: Whether to use caching (default: True)
        """
        self.cache = cache if cache else DataCache()
        self.use_cache = use_cache
        self._last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """Implement rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and error handling.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.RequestException: If request fails after retries
        """
        self._rate_limit()
        
        retry_count = 0
        last_exception = None
        
        while retry_count < self.MAX_RETRIES:
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                
                # Validate JSON response
                try:
                    data = response.json()
                    # Validate basic structure
                    if not isinstance(data, dict):
                        raise ValueError("API response is not a valid JSON object")
                    if 'MRData' not in data:
                        raise ValueError("API response missing 'MRData' field")
                    return data
                except ValueError as e:
                    logger.error(f"Invalid JSON response from API: {e}")
                    raise requests.RequestException(f"Invalid API response format: {e}")
                
            except requests.Timeout as e:
                last_exception = e
                retry_count += 1
                logger.warning(f"Request timeout (attempt {retry_count}/{self.MAX_RETRIES}): {url}")
                if retry_count < self.MAX_RETRIES:
                    # Exponential backoff
                    time.sleep(2 ** retry_count)
                    
            except requests.HTTPError as e:
                # Don't retry on client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    logger.error(f"Client error {e.response.status_code}: {url}")
                    raise
                # Retry on server errors (5xx)
                last_exception = e
                retry_count += 1
                logger.warning(f"Server error (attempt {retry_count}/{self.MAX_RETRIES}): {url} - {e}")
                if retry_count < self.MAX_RETRIES:
                    time.sleep(2 ** retry_count)
                    
            except requests.RequestException as e:
                last_exception = e
                retry_count += 1
                logger.warning(f"Request failed (attempt {retry_count}/{self.MAX_RETRIES}): {url} - {e}")
                if retry_count < self.MAX_RETRIES:
                    # Exponential backoff
                    time.sleep(2 ** retry_count)
        
        # All retries failed
        logger.error(f"Request failed after {self.MAX_RETRIES} attempts: {url}")
        raise last_exception
    
    def _get_cached_or_fetch(
        self,
        cache_key: str,
        url: str,
        ttl: int,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get data from cache or fetch from API.
        
        Args:
            cache_key: Cache key identifier
            url: API endpoint URL
            ttl: Cache TTL in seconds
            params: Query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.RequestException: If API request fails and no cache available
        """
        # Check cache first if enabled
        if self.use_cache:
            try:
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    logger.info(f"Cache hit: {cache_key}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Cache read error for {cache_key}: {e}")
                # Continue to fetch from API
        
        # Fetch from API
        logger.info(f"Fetching from API: {url}")
        try:
            data = self._make_request(url, params)
            
            # Store in cache if enabled
            if self.use_cache:
                try:
                    self.cache.set(cache_key, data, ttl)
                except Exception as e:
                    logger.warning(f"Cache write error for {cache_key}: {e}")
                    # Continue without caching
            
            return data
            
        except requests.RequestException as e:
            # If API fails, try to use stale cache as fallback
            if self.use_cache:
                logger.warning("API request failed, attempting to use stale cache...")
                try:
                    stale_data = self.cache.get(cache_key, ignore_ttl=True)
                    if stale_data is not None:
                        logger.info(f"Using stale cache data for {cache_key}")
                        return stale_data
                except Exception as cache_error:
                    logger.error(f"Failed to retrieve stale cache: {cache_error}")
            
            # No cache available, re-raise the original exception
            logger.error(f"No fallback data available for {cache_key}")
            raise

    def get_next_race(self) -> Race:
        """
        Get next scheduled F1 race information.
        
        Returns:
            Race object with next race details
            
        Raises:
            ValueError: If no upcoming race found or data is invalid
            requests.RequestException: If API request fails
        """
        cache_key = "next_race"
        url = f"{self.BASE_URL}/current/next.json"
        
        try:
            data = self._get_cached_or_fetch(cache_key, url, self.CACHE_TTL_NEXT_RACE)
        except requests.RequestException as e:
            logger.error(f"Failed to fetch next race: {e}")
            raise ValueError(f"Unable to fetch next race information: {e}")
        
        # Validate and parse response
        try:
            if 'MRData' not in data:
                raise ValueError("Invalid API response structure: missing 'MRData'")
            
            race_table = data['MRData'].get('RaceTable', {})
            races = race_table.get('Races', [])
            
            if not races:
                raise ValueError("No upcoming races found in API response")
            
            race_data = races[0]
            
            # Validate required fields
            required_fields = ['season', 'round', 'raceName', 'Circuit', 'date']
            missing_fields = [f for f in required_fields if f not in race_data]
            if missing_fields:
                raise ValueError(f"Race data missing required fields: {', '.join(missing_fields)}")
            
            return self._parse_race(race_data)
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse next race data: {e}")
            raise ValueError(f"Invalid race data structure: {e}")
    
    def get_current_season_results(self, season: Optional[int] = None) -> List[RaceResult]:
        """
        Get all race results for current or specified season.
        
        Args:
            season: Season year (uses current season if None)
            
        Returns:
            List of RaceResult objects (empty list if no results available)
            
        Raises:
            requests.RequestException: If API request fails
        """
        season_str = str(season) if season else "current"
        cache_key = f"season_results_{season_str}"
        url = f"{self.BASE_URL}/{season_str}/results.json"
        
        try:
            data = self._get_cached_or_fetch(
                cache_key,
                url,
                self.CACHE_TTL_SEASON_RESULTS,
                params={"limit": 1000}
            )
        except requests.RequestException as e:
            logger.error(f"Failed to fetch season results: {e}")
            return []  # Return empty list on error
        
        # Parse response
        results = []
        try:
            if 'MRData' not in data:
                logger.warning("Invalid API response: missing 'MRData'")
                return []
            
            race_table = data['MRData'].get('RaceTable', {})
            races = race_table.get('Races', [])
            
            for race_data in races:
                try:
                    race = self._parse_race(race_data)
                    for result_data in race_data.get('Results', []):
                        try:
                            result = self._parse_race_result(race, result_data)
                            results.append(result)
                        except (KeyError, ValueError, TypeError) as e:
                            logger.warning(f"Failed to parse race result: {e}")
                            continue
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse race data: {e}")
                    continue
                    
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to parse season results structure: {e}")
        
        return results
    
    def get_driver_standings(self, season: Optional[int] = None) -> List[DriverStanding]:
        """
        Get current driver championship standings.
        
        Args:
            season: Season year (uses current season if None)
            
        Returns:
            List of DriverStanding objects (empty list if unavailable)
            
        Raises:
            requests.RequestException: If API request fails
        """
        season_str = str(season) if season else "current"
        cache_key = f"driver_standings_{season_str}"
        url = f"{self.BASE_URL}/{season_str}/driverStandings.json"
        
        try:
            data = self._get_cached_or_fetch(cache_key, url, self.CACHE_TTL_STANDINGS)
        except requests.RequestException as e:
            logger.error(f"Failed to fetch driver standings: {e}")
            return []  # Return empty list on error
        
        # Parse response
        standings = []
        try:
            if 'MRData' not in data:
                logger.warning("Invalid API response: missing 'MRData'")
                return []
            
            standings_table = data['MRData'].get('StandingsTable', {})
            standings_lists = standings_table.get('StandingsLists', [])
            
            if not standings_lists:
                logger.warning("No standings data available")
                return []
            
            standings_list = standings_lists[0]
            driver_standings = standings_list.get('DriverStandings', [])
            
            for standing_data in driver_standings:
                try:
                    standing = self._parse_driver_standing(standing_data)
                    standings.append(standing)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse driver standing: {e}")
                    continue
                    
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse driver standings structure: {e}")
        
        return standings
    
    def get_constructor_standings(self, season: Optional[int] = None) -> List[ConstructorStanding]:
        """
        Get current constructor championship standings.
        
        Args:
            season: Season year (uses current season if None)
            
        Returns:
            List of ConstructorStanding objects (empty list if unavailable)
            
        Raises:
            requests.RequestException: If API request fails
        """
        season_str = str(season) if season else "current"
        cache_key = f"constructor_standings_{season_str}"
        url = f"{self.BASE_URL}/{season_str}/constructorStandings.json"
        
        try:
            data = self._get_cached_or_fetch(cache_key, url, self.CACHE_TTL_STANDINGS)
        except requests.RequestException as e:
            logger.error(f"Failed to fetch constructor standings: {e}")
            return []  # Return empty list on error
        
        # Parse response
        standings = []
        try:
            if 'MRData' not in data:
                logger.warning("Invalid API response: missing 'MRData'")
                return []
            
            standings_table = data['MRData'].get('StandingsTable', {})
            standings_lists = standings_table.get('StandingsLists', [])
            
            if not standings_lists:
                logger.warning("No constructor standings data available")
                return []
            
            standings_list = standings_lists[0]
            constructor_standings = standings_list.get('ConstructorStandings', [])
            
            for standing_data in constructor_standings:
                try:
                    standing = self._parse_constructor_standing(standing_data)
                    standings.append(standing)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse constructor standing: {e}")
                    continue
                    
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse constructor standings structure: {e}")
        
        return standings
    
    def get_qualifying_results(self, season: int, round_num: int) -> List[QualifyingResult]:
        """
        Get qualifying results for specific race.
        
        Args:
            season: Season year
            round_num: Race round number
            
        Returns:
            List of QualifyingResult objects (empty list if unavailable)
            
        Raises:
            requests.RequestException: If API request fails
        """
        cache_key = f"qualifying_{season}_{round_num}"
        url = f"{self.BASE_URL}/{season}/{round_num}/qualifying.json"
        
        try:
            data = self._get_cached_or_fetch(cache_key, url, self.CACHE_TTL_QUALIFYING)
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch qualifying results: {e}")
            return []  # Return empty list - qualifying may not be available yet
        
        # Parse response
        results = []
        try:
            if 'MRData' not in data:
                logger.warning("Invalid API response: missing 'MRData'")
                return []
            
            race_table = data['MRData'].get('RaceTable', {})
            races = race_table.get('Races', [])
            
            if races:
                race_data = races[0]
                try:
                    race = self._parse_race(race_data)
                    qualifying_results = race_data.get('QualifyingResults', [])
                    
                    for qual_data in qualifying_results:
                        try:
                            result = self._parse_qualifying_result(race, qual_data)
                            results.append(result)
                        except (KeyError, ValueError, TypeError) as e:
                            logger.warning(f"Failed to parse qualifying result: {e}")
                            continue
                            
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse race data: {e}")
                    
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to parse qualifying results structure: {e}")
        
        return results
    
    def get_circuit_history(self, circuit_id: str, years: int = 5) -> List[RaceResult]:
        """
        Get historical results for specific circuit.
        
        Args:
            circuit_id: Circuit identifier (e.g., 'monaco', 'silverstone')
            years: Number of years of history to fetch (default: 5)
            
        Returns:
            List of RaceResult objects (empty list if unavailable)
            
        Raises:
            requests.RequestException: If API request fails
        """
        cache_key = f"circuit_history_{circuit_id}_{years}"
        url = f"{self.BASE_URL}/circuits/{circuit_id}/results.json"
        
        try:
            data = self._get_cached_or_fetch(
                cache_key,
                url,
                self.CACHE_TTL_CIRCUIT_HISTORY,
                params={"limit": years * 30}  # Approximate limit
            )
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch circuit history: {e}")
            return []  # Return empty list - circuit history is optional
        
        # Parse response
        results = []
        try:
            if 'MRData' not in data:
                logger.warning("Invalid API response: missing 'MRData'")
                return []
            
            race_table = data['MRData'].get('RaceTable', {})
            races = race_table.get('Races', [])
            
            # Get only the most recent 'years' worth of races
            current_year = datetime.now().year
            cutoff_year = current_year - years
            
            for race_data in races:
                try:
                    race = self._parse_race(race_data)
                    if race.season >= cutoff_year:
                        for result_data in race_data.get('Results', []):
                            try:
                                result = self._parse_race_result(race, result_data)
                                results.append(result)
                            except (KeyError, ValueError, TypeError) as e:
                                logger.warning(f"Failed to parse race result: {e}")
                                continue
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse race data: {e}")
                    continue
                    
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to parse circuit history structure: {e}")
        
        return results

    # Helper methods for parsing API responses
    
    def _parse_circuit(self, circuit_data: Dict[str, Any]) -> Circuit:
        """Parse circuit data from API response."""
        required_fields = ['circuitId', 'circuitName', 'Location']
        for field in required_fields:
            if field not in circuit_data:
                raise ValueError(f"Circuit data missing required field: {field}")
        
        location = circuit_data['Location']
        if 'locality' not in location or 'country' not in location:
            raise ValueError("Circuit location data incomplete")
        
        return Circuit(
            circuit_id=circuit_data['circuitId'],
            circuit_name=circuit_data['circuitName'],
            location=location['locality'],
            country=location['country']
        )
    
    def _parse_driver(self, driver_data: Dict[str, Any]) -> Driver:
        """Parse driver data from API response."""
        required_fields = ['driverId', 'givenName', 'familyName', 'nationality']
        for field in required_fields:
            if field not in driver_data:
                raise ValueError(f"Driver data missing required field: {field}")
        
        return Driver(
            driver_id=driver_data['driverId'],
            code=driver_data.get('code', '???'),  # Some drivers may not have a code
            forename=driver_data['givenName'],
            surname=driver_data['familyName'],
            nationality=driver_data['nationality']
        )
    
    def _parse_constructor(self, constructor_data: Dict[str, Any]) -> Constructor:
        """Parse constructor data from API response."""
        required_fields = ['constructorId', 'name', 'nationality']
        for field in required_fields:
            if field not in constructor_data:
                raise ValueError(f"Constructor data missing required field: {field}")
        
        return Constructor(
            constructor_id=constructor_data['constructorId'],
            name=constructor_data['name'],
            nationality=constructor_data['nationality']
        )
    
    def _parse_race(self, race_data: Dict[str, Any]) -> Race:
        """Parse race data from API response."""
        required_fields = ['season', 'round', 'raceName', 'Circuit', 'date']
        for field in required_fields:
            if field not in race_data:
                raise ValueError(f"Race data missing required field: {field}")
        
        # Parse date and time
        try:
            date_str = race_data['date']
            time_str = race_data.get('time', '00:00:00Z')
            datetime_str = f"{date_str}T{time_str.rstrip('Z')}"
            race_date = datetime.fromisoformat(datetime_str)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid race date/time format: {e}")
        
        return Race(
            season=int(race_data['season']),
            round=int(race_data['round']),
            race_name=race_data['raceName'],
            circuit=self._parse_circuit(race_data['Circuit']),
            date=race_date
        )
    
    def _parse_race_result(self, race: Race, result_data: Dict[str, Any]) -> RaceResult:
        """Parse race result data from API response."""
        required_fields = ['Driver', 'Constructor', 'position', 'points', 'grid', 'laps', 'status']
        for field in required_fields:
            if field not in result_data:
                raise ValueError(f"Race result data missing required field: {field}")
        
        try:
            return RaceResult(
                race=race,
                driver=self._parse_driver(result_data['Driver']),
                constructor=self._parse_constructor(result_data['Constructor']),
                position=int(result_data['position']),
                points=float(result_data['points']),
                grid=int(result_data['grid']),
                laps=int(result_data['laps']),
                status=result_data['status']
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid race result data types: {e}")
    
    def _parse_qualifying_result(self, race: Race, qual_data: Dict[str, Any]) -> QualifyingResult:
        """Parse qualifying result data from API response."""
        required_fields = ['Driver', 'Constructor', 'position']
        for field in required_fields:
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
                q3_time=qual_data.get('Q3')
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid qualifying result data types: {e}")
    
    def _parse_driver_standing(self, standing_data: Dict[str, Any]) -> DriverStanding:
        """Parse driver standing data from API response."""
        required_fields = ['Driver', 'Constructors', 'position', 'points', 'wins']
        for field in required_fields:
            if field not in standing_data:
                raise ValueError(f"Driver standing data missing required field: {field}")
        
        if not standing_data['Constructors']:
            raise ValueError("Driver standing missing constructor information")
        
        try:
            return DriverStanding(
                driver=self._parse_driver(standing_data['Driver']),
                constructor=self._parse_constructor(standing_data['Constructors'][0]),
                position=int(standing_data['position']),
                points=float(standing_data['points']),
                wins=int(standing_data['wins'])
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid driver standing data types: {e}")
    
    def _parse_constructor_standing(self, standing_data: Dict[str, Any]) -> ConstructorStanding:
        """Parse constructor standing data from API response."""
        required_fields = ['Constructor', 'position', 'points', 'wins']
        for field in required_fields:
            if field not in standing_data:
                raise ValueError(f"Constructor standing data missing required field: {field}")
        
        try:
            return ConstructorStanding(
                constructor=self._parse_constructor(standing_data['Constructor']),
                position=int(standing_data['position']),
                points=float(standing_data['points']),
                wins=int(standing_data['wins'])
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid constructor standing data types: {e}")
