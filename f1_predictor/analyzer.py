"""
Statistical prediction analyzer.

Combines weighted 0-100 factor scores into a single score per driver, then
turns the scores into win probabilities (softmax) and podium probabilities
(Plackett-Luce). Probabilities are normalized across the field, so win
probabilities sum to 1 and podium probabilities sum to 3.
"""

import logging
from typing import Dict, List, Optional

from f1_predictor.features import (
    DriverFeatures, FACTOR_LABELS, RECENT_RACES_COUNT, RELIABILITY_WINDOW,
    driver_recent_results, extract_all, teammate_delta,
)
from f1_predictor.models import DriverPrediction, RaceContext
from f1_predictor.probability import plackett_luce_top3, softmax

logger = logging.getLogger(__name__)


class PredictionAnalyzer:
    """Rule-based analyzer with fixed factor weights."""

    name = "statistical"

    # Weights were chosen with the 2023-2026 walk-forward backtest (see
    # backtest.py). Grid position is by far the strongest single signal: with
    # qualifying at 20% the model picked the winner in 50% of races, at 45% it
    # picks 60%, and simply picking the pole sitter gets 63%.
    WEIGHTS: Dict[str, float] = {
        "championship": 0.15,
        "form": 0.12,
        "team": 0.10,
        "qualifying": 0.45,
        "circuit": 0.06,
        "reliability": 0.05,
        "teammate": 0.07,
    }

    # In the wet, grid position matters less and form/reliability matter more.
    WET_WEIGHT_OVERRIDES: Dict[str, float] = {
        "qualifying": 0.30,
        "form": 0.20,
        "reliability": 0.10,
    }

    # Softmax temperature applied to 0-100 scores. The backtest reports the
    # temperature with the lowest log loss (8 for 2023-2026); lower values make
    # predictions sharper.
    TEMPERATURE = 8.0

    def __init__(self, temperature: Optional[float] = None, weights: Optional[Dict[str, float]] = None):
        self.temperature = temperature if temperature is not None else self.TEMPERATURE
        self.weights = dict(weights) if weights else dict(self.WEIGHTS)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def active_weights(self, context: RaceContext) -> Dict[str, float]:
        weights = dict(self.weights)
        if context.weather is not None and context.weather.is_wet:
            weights.update(self.WET_WEIGHT_OVERRIDES)
        return weights

    def combine_factors(self, factors: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """Weighted average of the factor scores present (weights renormalized)."""
        if not factors:
            return 50.0
        weights = weights or self.weights
        total_score = 0.0
        total_weight = 0.0
        for name, score in factors.items():
            w = weights.get(name, 0.0)
            total_score += score * w
            total_weight += w
        if total_weight <= 0:
            return 50.0
        return max(0.0, min(100.0, total_score / total_weight))

    def score_features(self, features: DriverFeatures, context: RaceContext) -> Dict[str, float]:
        """Select the factors that feed the weighted score for this driver."""
        factors = dict(features.factors)
        if not context.has_qualifying:
            # No grid yet: use the sprint result as the best available proxy,
            # otherwise let the weights renormalize without it.
            if features.sprint_position:
                factors["qualifying"] = factors["sprint"]
            else:
                factors.pop("qualifying", None)
        factors.pop("sprint", None)
        return factors

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    def generate_reasoning(self, features: DriverFeatures, context: RaceContext, factors: Dict[str, float]) -> List[str]:
        driver = features.entry.driver
        constructor = features.entry.constructor
        reasoning: List[str] = []

        standing = next((s for s in context.driver_standings if s.driver.driver_id == driver.driver_id), None)
        if standing is not None and "championship" in factors:
            suffix = f" (end of {context.race.season - 1})" if context.standings_from_previous_season else ""
            reasoning.append(
                f"Championship: P{standing.position}{suffix}, {standing.points:g} pts, {standing.wins} win(s) "
                f"[{factors['championship']:.0f}/100]"
            )

        recent = driver_recent_results(driver.driver_id, context.recent_results, RECENT_RACES_COUNT)
        if recent and "form" in factors:
            wins = sum(1 for r in recent if r.position == 1)
            podiums = sum(1 for r in recent if r.position <= 3)
            avg = sum(r.position for r in recent) / len(recent)
            if wins:
                detail = f"{wins} win(s), avg P{avg:.1f}"
            elif podiums:
                detail = f"{podiums} podium(s), avg P{avg:.1f}"
            else:
                detail = f"avg P{avg:.1f}"
            reasoning.append(f"Recent form: {detail} in last {len(recent)} races [{factors['form']:.0f}/100]")

        if "team" in factors:
            team_standing = next((s for s in context.constructor_standings if s.constructor.constructor_id == constructor.constructor_id), None)
            pos = f"P{team_standing.position} in constructors" if team_standing else "no standing yet"
            reasoning.append(f"Team: {constructor.name}, {pos} [{factors['team']:.0f}/100]")

        if features.qualifying_position:
            label = "Pole position" if features.qualifying_position == 1 else f"P{features.qualifying_position} on the grid"
            reasoning.append(f"Qualifying: {label} [{factors.get('qualifying', features.values['qualifying']):.0f}/100]")
        elif features.sprint_position:
            reasoning.append(f"Sprint: finished P{features.sprint_position} (used in place of qualifying) [{features.values['sprint']:.0f}/100]")
        elif features.sprint_position is None and context.race.is_sprint:
            reasoning.append("Sprint weekend: sprint result not yet available")

        if features.sprint_position and features.qualifying_position:
            reasoning.append(f"Sprint: finished P{features.sprint_position}")

        mine_at_circuit = [r for r in context.circuit_history if r.driver.driver_id == driver.driver_id]
        if mine_at_circuit and "circuit" in factors:
            wins = sum(1 for r in mine_at_circuit if r.position == 1)
            podiums = sum(1 for r in mine_at_circuit if r.position <= 3)
            if wins:
                detail = f"{wins} win(s) in {len(mine_at_circuit)} races here"
            elif podiums:
                detail = f"{podiums} podium(s) in {len(mine_at_circuit)} races here"
            else:
                detail = f"no podium in {len(mine_at_circuit)} races here"
            reasoning.append(f"Circuit history: {detail} [{factors['circuit']:.0f}/100]")
        elif "circuit" in factors and context.circuit_history:
            reasoning.append("Circuit history: first race at this circuit")

        window = driver_recent_results(driver.driver_id, context.recent_results, RELIABILITY_WINDOW)
        if window and "reliability" in factors:
            dnfs = sum(1 for r in window if not r.finished)
            reasoning.append(f"Reliability: {dnfs} retirement(s) in last {len(window)} races [{factors['reliability']:.0f}/100]")

        delta = teammate_delta(driver.driver_id, constructor.constructor_id, context.recent_results)
        if delta is not None and "teammate" in factors:
            if abs(delta) < 0.5:
                detail = "level with teammate"
            elif delta > 0:
                detail = f"beats teammate by {delta:.1f} places on average"
            else:
                detail = f"behind teammate by {-delta:.1f} places on average"
            reasoning.append(f"Teammate: {detail} [{factors['teammate']:.0f}/100]")

        if context.weather is not None and context.weather.is_wet:
            reasoning.append("Weather: wet race expected, grid position weighted less")

        return reasoning

    def score_context(self, context: RaceContext) -> List[float]:
        """Raw 0-100 scores for every entry, in entry order (used for calibration)."""
        weights = self.active_weights(context)
        return [
            self.combine_factors(self.score_features(f, context), weights)
            for f in extract_all(context)
        ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, context: RaceContext, top_n: Optional[int] = None) -> List[DriverPrediction]:
        """
        Predict the race described by the context.

        Returns predictions for every entrant sorted by win probability. If
        top_n is given only the first top_n are returned.
        """
        if not context.entries:
            logger.error("No entries for %s", context.race.race_name)
            return []

        weights = self.active_weights(context)
        all_features = extract_all(context)
        scored = []
        for features in all_features:
            factors = self.score_features(features, context)
            score = self.combine_factors(factors, weights)
            scored.append((features, factors, score))

        win_probs = softmax([s for _, _, s in scored], temperature=self.temperature)
        podium_probs = plackett_luce_top3(win_probs)

        predictions: List[DriverPrediction] = []
        for (features, factors, score), p_win, p_podium in zip(scored, win_probs, podium_probs):
            display_factors = dict(features.factors)
            reasoning = self.generate_reasoning(features, context, factors)
            predictions.append(
                DriverPrediction(
                    driver=features.entry.driver,
                    constructor=features.entry.constructor,
                    win_probability=p_win,
                    podium_probability=p_podium,
                    score=score,
                    factors=display_factors,
                    reasoning=reasoning,
                    grid_position=features.qualifying_position,
                )
            )

        predictions.sort(key=lambda p: p.win_probability, reverse=True)
        logger.info("%s: %d predictions for %s", self.name, len(predictions), context.race.race_name)
        return predictions[:top_n] if top_n else predictions


def factor_label(name: str) -> str:
    return FACTOR_LABELS.get(name, name.replace("_", " ").title())
