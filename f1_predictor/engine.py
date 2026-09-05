"""
Prediction engine orchestrator.

Wires together data fetching, point-in-time context building, the analyzers
and the formatter. Used by the CLI and the Streamlit app.
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import requests

from f1_predictor.analyzer import PredictionAnalyzer
from f1_predictor.cache import DataCache
from f1_predictor.data_fetcher import F1DataFetcher
from f1_predictor.formatter import ResultFormatter, result_to_dict
from f1_predictor.history import HistoricalDataStore
from f1_predictor.models import DriverPrediction, PredictionError, PredictionResult, Race, RaceContext
from f1_predictor.weather import WeatherClient

logger = logging.getLogger(__name__)

DATA_SOURCE_F1 = "Jolpica F1 API"
DATA_SOURCE_WEATHER = "Open-Meteo"
MODEL_TITLES = {"statistical": "Statistical model", "ml": "ML model", "ai": "AI analyst", "verdict": "AI verdict"}


def blend_results(context: RaceContext, results: Dict[str, PredictionResult], top_n: Optional[int] = None) -> PredictionResult:
    """Average the win and podium probabilities of several full-field results."""
    from f1_predictor.probability import normalize, scale_to_sum

    names = [n for n in results if n != "verdict"]
    if not names:
        raise PredictionError("AnalysisError", "No model available to build a verdict", [], False)
    by_driver: Dict[str, Dict[str, DriverPrediction]] = {}
    for name in names:
        for p in results[name].predictions:
            by_driver.setdefault(p.driver.driver_id, {})[name] = p
    ids = list(by_driver.keys())
    win = normalize([sum(by_driver[d][n].win_probability for n in by_driver[d]) / len(by_driver[d]) for d in ids])
    podium = scale_to_sum([sum(by_driver[d][n].podium_probability for n in by_driver[d]) / len(by_driver[d]) for d in ids], target=3.0, cap=1.0)

    components = {d: {n: by_driver[d][n].win_probability for n in by_driver[d]} for d in ids}
    predictions: List[DriverPrediction] = []
    for d, p_win, p_podium in zip(ids, win, podium):
        sample = next(iter(by_driver[d].values()))
        parts = ", ".join(f"{MODEL_TITLES.get(n, n)} {by_driver[d][n].win_probability * 100:.1f}%" for n in by_driver[d])
        reasoning = [f"Blend of: {parts}"]
        ai_note = next((r for r in by_driver[d].get("ai", sample).reasoning if r.startswith("AI analyst:")), None)
        if ai_note:
            reasoning.append(ai_note)
        reasoning.extend(r for r in sample.reasoning if not r.startswith(("AI analyst:", "Model win probability", "Blend of")))
        predictions.append(DriverPrediction(
            driver=sample.driver, constructor=sample.constructor,
            win_probability=p_win, podium_probability=p_podium, score=p_win * 100,
            factors=dict(sample.factors), reasoning=reasoning, grid_position=sample.grid_position,
        ))
    predictions.sort(key=lambda p: p.win_probability, reverse=True)

    first = results[names[0]]
    ai_result = results.get("ai")
    notes = [f"Verdict blends {len(names)} model(s): " + ", ".join(MODEL_TITLES.get(n, n) for n in names)]
    if "ai" not in names:
        notes.append("AI analyst not included (unavailable)")
    return PredictionResult(
        race=first.race,
        predictions=predictions[:top_n] if top_n else predictions,
        generated_at=datetime.now(timezone.utc),
        data_sources=first.data_sources + (["Claude"] if ai_result else []),
        data_completeness=first.data_completeness,
        model_name="verdict",
        weather=first.weather,
        qualifying_available=first.qualifying_available,
        notes=notes + [n for n in first.notes],
        actual_results=first.actual_results,
        analysis=ai_result.analysis if ai_result else None,
        components=components,
    )


class PredictionEngine:
    """Orchestrates the F1 race prediction process."""

    def __init__(
        self,
        use_cache: bool = True,
        cache_dir: str = ".f1_cache",
        top_n: Optional[int] = 3,
        verbose: bool = False,
        use_ml: bool = False,
        model_path: Optional[str] = None,
        use_weather: bool = True,
        history_years: int = 5,
    ):
        self.use_cache = use_cache
        self.top_n = top_n
        self.verbose = verbose
        self.use_ml = use_ml
        self.model_path = model_path
        self.use_weather = use_weather

        self.cache = DataCache(cache_dir)
        self.data_fetcher = F1DataFetcher(cache=self.cache, use_cache=use_cache)
        self.weather_client = WeatherClient(cache=self.cache, use_cache=use_cache) if use_weather else None
        self.store = HistoricalDataStore(self.data_fetcher, self.weather_client, history_years=history_years)
        self.formatter = ResultFormatter()

        self._analyzers: Dict[str, PredictionAnalyzer] = {"statistical": PredictionAnalyzer()}
        self.analyzer = self.get_analyzer("ml" if use_ml else "statistical")
        logger.info("Prediction engine initialized (model=%s, cache=%s)", self.analyzer.name, use_cache)

    # ------------------------------------------------------------------
    # Analyzers
    # ------------------------------------------------------------------

    def get_analyzer(self, name: str) -> PredictionAnalyzer:
        if name not in self._analyzers:
            if name == "ml":
                from f1_predictor.ml_analyzer import MLPredictionAnalyzer
                self._analyzers[name] = MLPredictionAnalyzer(model_path=self.model_path)
            elif name == "ai":
                from f1_predictor.ai_analyzer import AIPredictionAnalyzer
                self._analyzers[name] = AIPredictionAnalyzer(
                    cache=self.cache, use_cache=self.use_cache,
                    reference_analyzers=[self.get_analyzer("statistical"), self.get_analyzer("ml")],
                )
            else:
                raise ValueError(f"Unknown model: {name}")
        return self._analyzers[name]

    @property
    def ml_available(self) -> bool:
        return getattr(self.get_analyzer("ml"), "model_loaded", False)

    @property
    def ai_available(self) -> bool:
        return getattr(self.get_analyzer("ai"), "model_loaded", False)

    def availability_error(self, name: str) -> Optional[str]:
        return getattr(self.get_analyzer(name), "load_error", None)

    # ------------------------------------------------------------------
    # Race lookup
    # ------------------------------------------------------------------

    def _show_progress(self, message: str) -> None:
        if self.verbose:
            print(f"[*] {message}", file=sys.stderr)
        logger.info(message)

    def get_next_race(self) -> Race:
        try:
            return self.data_fetcher.get_next_race()
        except ValueError as e:
            raise PredictionError(
                error_type="DataError",
                message=f"Failed to retrieve next race information: {e}",
                suggestions=["Check if the F1 season is currently active", "Try again later"],
                recoverable=True,
            )
        except requests.RequestException as e:
            raise self._network_error(e)

    def current_season(self) -> int:
        return self.data_fetcher.get_current_season()

    def get_schedule(self, season: int) -> List[Race]:
        try:
            return self.store.schedule(season)
        except requests.RequestException as e:
            raise self._network_error(e)

    def completed_rounds(self, season: int) -> List[int]:
        return self.store.completed_rounds(season)

    def resolve_race(self, selector: Union[int, str], season: Optional[int] = None) -> Race:
        """
        Find a race by round number or by a name fragment.

        Matches the race name, circuit name, circuit id, locality or country,
        case-insensitively.
        """
        season = season or self.current_season()
        schedule = self.get_schedule(season)
        if not schedule:
            raise PredictionError("DataError", f"No schedule found for {season}", ["Check the season year"], True)

        text = str(selector).strip()
        if text.isdigit():
            rnd = int(text)
            for race in schedule:
                if race.round == rnd:
                    return race
            raise PredictionError("DataError", f"Round {rnd} does not exist in {season} ({len(schedule)} rounds)", [], True)

        needle = text.lower()
        matches = [
            r for r in schedule
            if needle in r.race_name.lower()
            or needle in r.circuit.circuit_name.lower()
            or needle in r.circuit.circuit_id.lower()
            or needle in r.circuit.location.lower()
            or needle in r.circuit.country.lower()
        ]
        if not matches:
            names = ", ".join(r.race_name for r in schedule)
            raise PredictionError("DataError", f"No {season} race matches '{selector}'", [f"Available races: {names}"], True)
        if len(matches) > 1:
            exact = [r for r in matches if r.race_name.lower() == needle]
            if len(exact) == 1:
                return exact[0]
            names = ", ".join(f"{r.race_name} (round {r.round})" for r in matches)
            raise PredictionError("DataError", f"'{selector}' matches several races: {names}", ["Use the round number instead"], True)
        return matches[0]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def build_context(self, season: int, round_num: int, include_actual: bool = True) -> RaceContext:
        self._show_progress(f"Loading data for {season} round {round_num}...")
        try:
            return self.store.build_context(season, round_num, include_actual=include_actual, use_weather=self.use_weather)
        except requests.RequestException as e:
            raise self._network_error(e)
        except ValueError as e:
            raise PredictionError("DataError", str(e), ["Check the season and round"], True)

    def predict_context(self, context: RaceContext, analyzer: Optional[PredictionAnalyzer] = None, top_n: Optional[int] = -1) -> PredictionResult:
        """Predict a race from a prepared context (top_n=-1 means use the engine default)."""
        analyzer = analyzer or self.analyzer
        top_n = self.top_n if top_n == -1 else top_n
        if not context.entries:
            raise PredictionError(
                error_type="MissingData",
                message=f"No entry list could be determined for {context.race.race_name}",
                suggestions=["The season may not have started yet", "Try again once qualifying has run"],
                recoverable=False,
            )
        self._show_progress(f"Analyzing {context.race.race_name} with the {analyzer.name} model...")
        predictions = analyzer.analyze(context)
        if not predictions:
            raise PredictionError("AnalysisError", "Failed to generate predictions", ["Check data quality"], False)

        sources = [DATA_SOURCE_F1]
        if context.weather is not None:
            sources.append(DATA_SOURCE_WEATHER)
        if analyzer.name == "ai" and getattr(analyzer, "model_loaded", False):
            sources.append(f"Claude ({getattr(analyzer, 'model', 'claude')})")
        notes = list(context.notes)
        loaded = getattr(analyzer, "model_loaded", True)
        if not loaded:
            notes.append(f"{MODEL_TITLES.get(analyzer.name, analyzer.name)} unavailable, statistical model used ({getattr(analyzer, 'load_error', None)})")
        model_name = analyzer.name if loaded else "statistical"
        analysis = getattr(analyzer, "last_analysis", None) if loaded else None

        return PredictionResult(
            race=context.race,
            predictions=predictions[:top_n] if top_n else predictions,
            generated_at=datetime.now(timezone.utc),
            data_sources=sources,
            data_completeness=self._data_completeness(context),
            model_name=model_name,
            weather=context.weather,
            qualifying_available=context.has_qualifying,
            notes=notes,
            actual_results=context.actual_results,
            analysis=analysis,
        )

    def predict_race(self, season: int, round_num: int, model: Optional[str] = None) -> PredictionResult:
        analyzer = self.get_analyzer(model) if model else self.analyzer
        context = self.build_context(season, round_num)
        return self.predict_context(context, analyzer)

    def predict_next_race(self, model: Optional[str] = None) -> PredictionResult:
        race = self.get_next_race()
        logger.info("Next race: %s (%s round %d)", race.race_name, race.season, race.round)
        return self.predict_race(race.season, race.round, model)

    def compare(self, season: int, round_num: int, models: Optional[List[str]] = None) -> Dict[str, PredictionResult]:
        """Run several models on the same context."""
        if models is None:
            models = ["statistical", "ml"] + (["ai"] if self.ai_available else [])
        context = self.build_context(season, round_num)
        return {name: self.predict_context(context, self.get_analyzer(name)) for name in models}

    def verdict(self, season: int, round_num: int) -> Dict[str, PredictionResult]:
        """
        Run every model and blend them into a final verdict.

        Returns the individual results plus a "verdict" entry whose win
        probabilities are the mean of the available models (renormalized).
        """
        context = self.build_context(season, round_num)
        results: Dict[str, PredictionResult] = {}
        for name in ("statistical", "ml", "ai"):
            analyzer = self.get_analyzer(name)
            if not getattr(analyzer, "model_loaded", True):
                continue
            results[name] = self.predict_context(context, analyzer, top_n=None)  # full field for blending
        results["verdict"] = blend_results(context, results, top_n=self.top_n)
        if self.top_n:
            for name in list(results):
                if name != "verdict":
                    results[name].predictions = results[name].predictions[: self.top_n]
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _data_completeness(context: RaceContext) -> float:
        checks = [
            bool(context.driver_standings),
            bool(context.recent_results),
            context.has_qualifying,
            bool(context.circuit_history),
            context.weather is not None,
        ]
        return sum(checks) / len(checks)

    @staticmethod
    def _network_error(e: Exception) -> PredictionError:
        return PredictionError(
            error_type="NetworkError",
            message=f"Network error while fetching data: {e}",
            suggestions=["Check your internet connection", "Try again later", "Cached data is used when available"],
            recoverable=True,
        )

    def refresh_season(self, season: int) -> int:
        """Drop cached data for a season so the next call refetches it."""
        removed = 0
        for prefix in (f"season_results_{season}", f"season_sprints_{season}", f"season_qualifying_{season}",
                       f"schedule_{season}", "next_race", f"weather_{season}_"):
            removed += self.cache.clear(prefix=prefix)
        self.store.invalidate(season)
        return removed

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_result(self, result: PredictionResult) -> str:
        return self.formatter.format_prediction(result, verbose=self.verbose)

    def format_comparison(self, results: Dict[str, PredictionResult]) -> str:
        return self.formatter.format_comparison(results, top_n=self.top_n or 10)

    def to_dict(self, result: PredictionResult) -> dict:
        return result_to_dict(result)

    def format_error(self, error: PredictionError) -> str:
        lines = ["═" * 65, f"ERROR: {error.error_type}", "═" * 65, "", error.message, ""]
        if error.suggestions:
            lines.append("Suggestions:")
            lines.extend(f"  • {s}" for s in error.suggestions)
        lines.append("═" * 65)
        return "\n".join(lines)
