"""
AI analyst: Claude reads the same pre-race briefing as the other models and
returns win/podium probabilities plus a written analysis.

Requires the `anthropic` package and credentials (ANTHROPIC_API_KEY, or an
`ant auth login` profile). Without them the analyzer reports itself as
unavailable and the engine falls back to the statistical model.

Responses are cached per race (and per qualifying state) so repeated views
cost nothing.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from f1_predictor.analyzer import PredictionAnalyzer
from f1_predictor.cache import DataCache
from f1_predictor.features import FACTOR_LABELS, RECENT_RACES_COUNT, RELIABILITY_WINDOW, driver_recent_results, extract_all, teammate_delta
from f1_predictor.models import DriverPrediction, RaceContext
from f1_predictor.probability import normalize, scale_to_sum

logger = logging.getLogger(__name__)

DEFAULT_AI_MODEL = os.environ.get("F1_AI_MODEL", "claude-opus-5")
CACHE_TTL = 6 * 3600
MIN_PROBABILITY = 0.001  # floor for drivers the model leaves out

SYSTEM_PROMPT = """You are a Formula 1 race analyst. You receive a pre-race briefing for one Grand Prix:
the grid, championship standings, each driver's recent form, circuit history, reliability,
teammate comparison, race-day weather, and the picks of two quantitative models (a weighted
statistical model and a logistic-regression model trained on races since 2020).

Give your own assessment of who wins and who reaches the podium. Use the quantitative picks as a
reference, but apply racing judgement they cannot: track characteristics (overtaking difficulty,
slipstream, power sensitivity, tyre wear), how a car-driver pairing suits the circuit, the effect of
a wet race, sprint-weekend dynamics, championship pressure, and any patterns you see in the form data.

Rules:
- Return a win probability and a podium probability (both in percent) for EVERY driver listed in
  the briefing, using their driver_id exactly as given.
- Win probabilities should sum to roughly 100 and podium probabilities to roughly 300.
- Be honest about uncertainty: in a competitive field no driver deserves more than about 60%.
- Keep the analysis to three short paragraphs: the favourite and why, the main threats, and the
  wildcards or risks. Refer to drivers by surname.
- One short note per driver in the top six explaining their number."""

OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "key_factors": {"type": "array", "items": {"type": "string"}},
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "driver_id": {"type": "string"},
                    "win_probability": {"type": "number"},
                    "podium_probability": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["driver_id", "win_probability", "podium_probability", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["analysis", "key_factors", "predictions"],
    "additionalProperties": False,
}


def credentials_available() -> bool:
    """True if the Anthropic SDK can find credentials without prompting."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    profile_dir = os.path.join(os.path.expanduser("~"), ".config", "anthropic")
    return os.path.isdir(profile_dir) and any(os.scandir(profile_dir))


class AIPredictionAnalyzer(PredictionAnalyzer):
    """Claude-backed analyzer with statistical fallback."""

    name = "ai"

    def __init__(
        self,
        model: str = DEFAULT_AI_MODEL,
        cache: Optional[DataCache] = None,
        use_cache: bool = True,
        client: Any = None,
        reference_analyzers: Optional[List[PredictionAnalyzer]] = None,
    ):
        super().__init__()
        self.model = model
        self.cache = cache
        self.use_cache = use_cache and cache is not None
        self._client = client
        self.reference_analyzers = reference_analyzers or []
        self.load_error: Optional[str] = None
        self.last_analysis: Optional[str] = None
        self.last_key_factors: List[str] = []
        self.last_usage: Optional[Dict[str, int]] = None
        if client is None:
            try:
                import anthropic  # noqa: F401
            except ImportError:
                self.load_error = "The `anthropic` package is not installed (pip install anthropic)."
            else:
                if not credentials_available():
                    self.load_error = "No Anthropic credentials found. Set ANTHROPIC_API_KEY or run `ant auth login`."

    @property
    def model_loaded(self) -> bool:
        return self.load_error is None

    available = model_loaded

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    # ------------------------------------------------------------------
    # Briefing
    # ------------------------------------------------------------------

    def build_briefing(self, context: RaceContext) -> Dict[str, Any]:
        race = context.race
        features = extract_all(context)
        standings = {s.driver.driver_id: s for s in context.driver_standings}
        constructor_pos = {s.constructor.constructor_id: s for s in context.constructor_standings}

        drivers = []
        for f in features:
            d, c = f.entry.driver, f.entry.constructor
            recent = driver_recent_results(d.driver_id, context.recent_results, RECENT_RACES_COUNT)
            window = driver_recent_results(d.driver_id, context.recent_results, RELIABILITY_WINDOW)
            at_circuit = [r for r in context.circuit_history if r.driver.driver_id == d.driver_id]
            s = standings.get(d.driver_id)
            cs = constructor_pos.get(c.constructor_id)
            delta = teammate_delta(d.driver_id, c.constructor_id, context.recent_results)
            drivers.append({
                "driver_id": d.driver_id,
                "name": d.full_name,
                "team": c.name,
                "grid": f.qualifying_position,
                "sprint_result": f.sprint_position,
                "championship": {"position": s.position, "points": s.points, "wins": s.wins} if s else None,
                "team_standing": cs.position if cs else None,
                "last_races": [
                    {"race": r.race.race_name, "season": r.race.season, "finish": r.position, "grid": r.grid, "status": r.status}
                    for r in recent
                ],
                "retirements_last_10": sum(1 for r in window if not r.finished),
                "circuit_history": [{"season": r.race.season, "finish": r.position, "grid": r.grid} for r in at_circuit],
                "teammate_gap": round(delta, 1) if delta is not None else None,
                "factor_scores": {FACTOR_LABELS.get(k, k): round(v) for k, v in f.factors.items()},
            })

        briefing: Dict[str, Any] = {
            "race": {
                "name": race.race_name,
                "season": race.season,
                "round": race.round,
                "total_rounds": race.total_rounds,
                "circuit": race.circuit.circuit_name,
                "location": f"{race.circuit.location}, {race.circuit.country}",
                "date": race.date.date().isoformat(),
                "sprint_weekend": race.is_sprint,
                "qualifying_available": context.has_qualifying,
            },
            "weather": context.weather.describe() if context.weather else "unknown",
            "notes": context.notes,
            "drivers": drivers,
        }

        references = {}
        for analyzer in self.reference_analyzers:
            try:
                preds = analyzer.analyze(context)
            except Exception as e:  # reference picks are optional
                logger.debug("Reference analyzer %s failed: %s", analyzer.name, e)
                continue
            references[analyzer.name] = [
                {"driver_id": p.driver.driver_id, "win": round(p.win_probability * 100, 1), "podium": round(p.podium_probability * 100, 1)}
                for p in preds[:8]
            ]
        if references:
            briefing["model_picks"] = references
        return briefing

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _cache_key(self, context: RaceContext) -> str:
        quali = "q" if context.has_qualifying else "noq"
        sprint = "s" if context.sprint_results else "nos"
        return f"ai_prediction_{context.race.season}_{context.race.round}_{quali}_{sprint}_{self.model}"

    def _ask_claude(self, briefing: Dict[str, Any]) -> Dict[str, Any]:
        import anthropic

        client = self._get_client()
        prompt = (
            "Pre-race briefing (JSON):\n\n" + json.dumps(briefing, indent=1, sort_keys=True)
            + "\n\nProduce your prediction in the required JSON format."
        )
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            )
        except anthropic.AuthenticationError as e:
            raise RuntimeError(f"Anthropic authentication failed: {e.message}")
        except anthropic.RateLimitError:
            raise RuntimeError("Anthropic rate limit reached, try again in a minute")
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Anthropic API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"Could not reach the Anthropic API: {e}")

        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined to answer this briefing")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("The model's answer was cut off; try again")

        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {"input_tokens": getattr(usage, "input_tokens", 0), "output_tokens": getattr(usage, "output_tokens", 0)}
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Model returned invalid JSON: {e}")

    def fetch_prediction(self, context: RaceContext) -> Dict[str, Any]:
        """Return the raw model answer, from cache when possible."""
        key = self._cache_key(context)
        if self.use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                logger.info("AI prediction served from cache for %s", context.race.race_name)
                return cached
        briefing = self.build_briefing(context)
        answer = self._ask_claude(briefing)
        if self.use_cache:
            self.cache.set(key, answer, CACHE_TTL)
        return answer

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    def analyze(self, context: RaceContext, top_n: Optional[int] = None) -> List[DriverPrediction]:
        if not self.model_loaded:
            logger.info("AI analyst unavailable (%s), using statistical analysis", self.load_error)
            return super().analyze(context, top_n)
        if not context.entries:
            return []

        try:
            answer = self.fetch_prediction(context)
        except Exception as e:
            self.load_error = str(e)
            logger.error("AI analyst failed: %s", e)
            return super().analyze(context, top_n)

        by_id = {p.get("driver_id"): p for p in answer.get("predictions", []) if isinstance(p, dict)}
        self.last_analysis = answer.get("analysis") or None
        self.last_key_factors = [str(k) for k in answer.get("key_factors", [])]

        features = extract_all(context)
        raw_win = [max(MIN_PROBABILITY, float(by_id.get(f.entry.driver.driver_id, {}).get("win_probability", 0)) / 100) for f in features]
        raw_podium = [max(MIN_PROBABILITY, float(by_id.get(f.entry.driver.driver_id, {}).get("podium_probability", 0)) / 100) for f in features]
        win_probs = normalize(raw_win)
        podium_probs = scale_to_sum(raw_podium, target=3.0, cap=1.0)

        predictions: List[DriverPrediction] = []
        for f, p_win, p_podium in zip(features, win_probs, podium_probs):
            note = by_id.get(f.entry.driver.driver_id, {}).get("note")
            factors = self.score_features(f, context)
            reasoning = self.generate_reasoning(f, context, factors)
            if note:
                reasoning.insert(0, f"AI analyst: {note}")
            predictions.append(DriverPrediction(
                driver=f.entry.driver,
                constructor=f.entry.constructor,
                win_probability=p_win,
                podium_probability=p_podium,
                score=p_win * 100,
                factors=dict(f.factors),
                reasoning=reasoning,
                grid_position=f.qualifying_position,
            ))
        predictions.sort(key=lambda p: p.win_probability, reverse=True)
        missing = len(features) - sum(1 for f in features if f.entry.driver.driver_id in by_id)
        if missing:
            logger.warning("AI analyst omitted %d driver(s); they were given a floor probability", missing)
        return predictions[:top_n] if top_n else predictions
