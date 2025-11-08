"""
Prediction engine orchestrator module.

Coordinates all components to generate F1 race winner predictions.
Handles data fetching, validation, analysis, and formatting.
"""

import logging
from datetime import datetime
from typing import Optional, List
import sys
import requests

from f1_predictor.models import (
    Race, PredictionResult, PredictionError,
    DriverStanding, ConstructorStanding, RaceResult, QualifyingResult
)
from f1_predictor.data_fetcher import F1DataFetcher
from f1_predictor.cache import DataCache
from f1_predictor.analyzer import PredictionAnalyzer
from f1_predictor.formatter import ResultFormatter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Orchestrates the F1 race prediction process.
    
    Coordinates data fetching, validation, analysis, and formatting
    to generate comprehensive race winner predictions.
    """
    
    def __init__(
        self,
        use_cache: bool = True,
        cache_dir: str = ".f1_cache",
        top_n: int = 3,
        verbose: bool = False
    ):
        """
        Initialize the prediction engine.
        
        Args:
            use_cache: Whether to use data caching (default: True)
            cache_dir: Directory for cache storage (default: ".f1_cache")
            top_n: Number of top predictions to generate (default: 3)
            verbose: Whether to show detailed output (default: False)
        """
        self.use_cache = use_cache
        self.top_n = top_n
        self.verbose = verbose
        
        # Initialize components
        self.cache = DataCache(cache_dir) if use_cache else None
        self.data_fetcher = F1DataFetcher(cache=self.cache, use_cache=use_cache)
        self.analyzer = PredictionAnalyzer()
        self.formatter = ResultFormatter()
        
        logger.info("Prediction engine initialized")
        logger.info(f"Cache enabled: {use_cache}")
        logger.info(f"Top predictions: {top_n}")

    def _show_progress(self, message: str) -> None:
        """
        Display progress indicator to user.
        
        Args:
            message: Progress message to display
        """
        if self.verbose:
            print(f"[*] {message}", file=sys.stderr)
        logger.info(message)
    
    def _validate_race_data(self, race: Race) -> List[str]:
        """
        Validate race data for completeness.
        
        Args:
            race: Race object to validate
            
        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []
        
        if not race.race_name:
            warnings.append("Race name is missing")
        
        if not race.circuit or not race.circuit.circuit_id:
            warnings.append("Circuit information is incomplete")
        
        if not race.date:
            warnings.append("Race date is missing")
        
        if race.season < 1950 or race.season > datetime.now().year + 1:
            warnings.append(f"Race season {race.season} seems invalid")
        
        return warnings
    
    def _validate_standings_data(
        self,
        driver_standings: List[DriverStanding],
        constructor_standings: List[ConstructorStanding]
    ) -> List[str]:
        """
        Validate standings data for completeness.
        
        Args:
            driver_standings: List of driver standings
            constructor_standings: List of constructor standings
            
        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []
        
        if not driver_standings:
            warnings.append("No driver standings data available")
        elif len(driver_standings) < 10:
            warnings.append(f"Limited driver standings data ({len(driver_standings)} drivers)")
        
        if not constructor_standings:
            warnings.append("No constructor standings data available")
        elif len(constructor_standings) < 5:
            warnings.append(f"Limited constructor standings data ({len(constructor_standings)} teams)")
        
        return warnings
    
    def _validate_results_data(self, results: List[RaceResult]) -> List[str]:
        """
        Validate race results data for completeness.
        
        Args:
            results: List of race results
            
        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []
        
        if not results:
            warnings.append("No race results data available")
        elif len(results) < 20:
            warnings.append(f"Limited race results data ({len(results)} results)")
        
        return warnings
    
    def _log_data_quality_issues(self, warnings: List[str]) -> None:
        """
        Log data quality issues.
        
        Args:
            warnings: List of validation warnings
        """
        if warnings:
            logger.warning("Data quality issues detected:")
            for warning in warnings:
                logger.warning(f"  - {warning}")

    def predict_next_race(self) -> PredictionResult:
        """
        Generate predictions for the next scheduled F1 race.
        
        Orchestrates the complete prediction process:
        1. Fetch next race information
        2. Fetch current season data (standings, results)
        3. Fetch qualifying results (if available)
        4. Fetch circuit history
        5. Validate all data
        6. Generate predictions using analyzer
        7. Return formatted results
        
        Returns:
            PredictionResult object with predictions and metadata
            
        Raises:
            PredictionError: If prediction cannot be generated
        """
        try:
            # Step 1: Get next race information
            self._show_progress("Fetching next race information...")
            try:
                race = self.data_fetcher.get_next_race()
            except ValueError as e:
                raise PredictionError(
                    error_type="DataError",
                    message=f"Failed to retrieve next race information: {str(e)}",
                    suggestions=[
                        "Check if the F1 season is currently active",
                        "Verify API connectivity",
                        "Check if there are any scheduled races",
                        "Try again later"
                    ],
                    recoverable=True
                )
            except requests.RequestException as e:
                raise PredictionError(
                    error_type="NetworkError",
                    message=f"Network error while fetching race information: {str(e)}",
                    suggestions=[
                        "Check your internet connection",
                        "Verify the API is accessible",
                        "Try again later",
                        "Use --no-cache flag to bypass cache"
                    ],
                    recoverable=True
                )
            
            # Validate race data
            race_warnings = self._validate_race_data(race)
            if race_warnings:
                self._log_data_quality_issues(race_warnings)
            
            logger.info(f"Next race: {race.race_name} at {race.circuit.circuit_name}")
            logger.info(f"Date: {race.date.strftime('%Y-%m-%d')}")
            
            # Step 2: Get current season standings
            self._show_progress("Fetching driver and constructor standings...")
            try:
                driver_standings = self.data_fetcher.get_driver_standings(race.season)
                constructor_standings = self.data_fetcher.get_constructor_standings(race.season)
            except requests.RequestException as e:
                raise PredictionError(
                    error_type="NetworkError",
                    message=f"Network error while fetching standings: {str(e)}",
                    suggestions=[
                        "Check your internet connection",
                        "Verify the API is accessible",
                        "Try again later",
                        "Use --no-cache flag to bypass cache"
                    ],
                    recoverable=True
                )
            except Exception as e:
                raise PredictionError(
                    error_type="DataError",
                    message=f"Failed to retrieve standings data: {str(e)}",
                    suggestions=[
                        "Check if the F1 season has started",
                        "Verify API is returning valid data",
                        "Try again later"
                    ],
                    recoverable=True
                )
            
            # Validate standings data
            standings_warnings = self._validate_standings_data(
                driver_standings, constructor_standings
            )
            if standings_warnings:
                self._log_data_quality_issues(standings_warnings)
            
            if not driver_standings:
                raise PredictionError(
                    error_type="MissingData",
                    message="Cannot generate predictions without driver standings data",
                    suggestions=[
                        "Check if the F1 season has started",
                        "Verify API connectivity",
                        "Try again later"
                    ],
                    recoverable=False
                )
            
            logger.info(f"Loaded {len(driver_standings)} driver standings")
            logger.info(f"Loaded {len(constructor_standings)} constructor standings")
            
            # Step 3: Get current season results for form calculation
            self._show_progress("Fetching current season race results...")
            try:
                season_results = self.data_fetcher.get_current_season_results(race.season)
            except requests.RequestException as e:
                logger.warning(f"Network error fetching season results: {e}")
                logger.warning("Predictions will be generated with limited form data")
                season_results = []
            except Exception as e:
                logger.warning(f"Failed to fetch season results: {e}")
                logger.warning("Predictions will be generated with limited form data")
                season_results = []
            
            # Validate results data
            results_warnings = self._validate_results_data(season_results)
            if results_warnings:
                self._log_data_quality_issues(results_warnings)
            
            logger.info(f"Loaded {len(season_results)} race results from current season")
            
            # Step 4: Get qualifying results (may not be available yet)
            self._show_progress("Fetching qualifying results...")
            qualifying_results = []
            try:
                qualifying_results = self.data_fetcher.get_qualifying_results(
                    race.season, race.round
                )
                if qualifying_results:
                    logger.info(f"Loaded {len(qualifying_results)} qualifying results")
                else:
                    logger.info("Qualifying results not yet available")
            except requests.RequestException as e:
                logger.warning(f"Network error fetching qualifying results: {e}")
                logger.info("Predictions will be generated without qualifying data")
            except Exception as e:
                logger.warning(f"Unexpected error fetching qualifying results: {e}")
                logger.info("Predictions will be generated without qualifying data")
            
            # Step 5: Get circuit history
            self._show_progress("Fetching circuit history...")
            circuit_history = []
            try:
                circuit_history = self.data_fetcher.get_circuit_history(
                    race.circuit.circuit_id, years=5
                )
                if circuit_history:
                    logger.info(f"Loaded {len(circuit_history)} historical results for circuit")
                else:
                    logger.info("No circuit history available")
            except requests.RequestException as e:
                logger.warning(f"Network error fetching circuit history: {e}")
                logger.info("Predictions will be generated without circuit history")
            except Exception as e:
                logger.warning(f"Unexpected error fetching circuit history: {e}")
                logger.info("Predictions will be generated without circuit history")
            
            # Step 6: Generate predictions
            self._show_progress("Analyzing data and generating predictions...")
            predictions = self.analyzer.analyze(
                race=race,
                driver_standings=driver_standings,
                constructor_standings=constructor_standings,
                recent_results=season_results,
                qualifying_results=qualifying_results,
                circuit_history=circuit_history,
                top_n=self.top_n
            )
            
            if not predictions:
                raise PredictionError(
                    error_type="AnalysisError",
                    message="Failed to generate predictions from available data",
                    suggestions=[
                        "Check data quality",
                        "Verify sufficient historical data exists",
                        "Try again later"
                    ],
                    recoverable=False
                )
            
            logger.info(f"Generated {len(predictions)} predictions")
            
            # Step 7: Calculate data completeness
            data_completeness = self._calculate_data_completeness(
                qualifying_results, circuit_history
            )
            
            # Step 8: Create prediction result
            result = PredictionResult(
                race=race,
                predictions=predictions,
                generated_at=datetime.utcnow(),
                data_sources=["Jolpica F1 API"],
                data_completeness=data_completeness
            )
            
            self._show_progress("Prediction generation complete!")
            
            return result
            
        except PredictionError:
            # Re-raise prediction errors as-is
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
            raise PredictionError(
                error_type="UnexpectedError",
                message=f"An unexpected error occurred: {str(e)}",
                suggestions=[
                    "Check your internet connection",
                    "Verify the API is accessible",
                    "Check logs for more details",
                    "Try again later"
                ],
                recoverable=True
            )

    def _calculate_data_completeness(
        self,
        qualifying_results: Optional[List[QualifyingResult]],
        circuit_history: Optional[List[RaceResult]]
    ) -> float:
        """
        Calculate data completeness factor.
        
        Tracks which data sources are available:
        - Driver standings (always required)
        - Constructor standings (always required)
        - Season results (always required)
        - Qualifying results (optional)
        - Circuit history (optional)
        
        Args:
            qualifying_results: Qualifying results if available
            circuit_history: Circuit history if available
            
        Returns:
            Data completeness factor from 0.0 to 1.0
        """
        total_sources = 5
        available_sources = 3  # Standings and season results always available
        
        if qualifying_results is not None and len(qualifying_results) > 0:
            available_sources += 1
        
        if circuit_history is not None and len(circuit_history) > 0:
            available_sources += 1
        
        return available_sources / total_sources
    
    def format_result(self, result: PredictionResult) -> str:
        """
        Format prediction result for display.
        
        Args:
            result: PredictionResult to format
            
        Returns:
            Formatted string ready for display
        """
        return self.formatter.format_prediction(result, verbose=self.verbose)
    
    def format_error(self, error: PredictionError) -> str:
        """
        Format prediction error for display.
        
        Args:
            error: PredictionError to format
            
        Returns:
            Formatted error message string
        """
        lines = []
        lines.append("═" * 65)
        lines.append(f"ERROR: {error.error_type}")
        lines.append("═" * 65)
        lines.append(f"\n{error.message}\n")
        
        if error.suggestions:
            lines.append("Suggestions:")
            for suggestion in error.suggestions:
                lines.append(f"  • {suggestion}")
        
        lines.append("\n" + "═" * 65)
        
        return "\n".join(lines)
