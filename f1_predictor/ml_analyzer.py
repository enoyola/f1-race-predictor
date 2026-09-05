"""
Machine-learning prediction analyzer.

Loads a trained model bundle (see training.py) and predicts win and podium
probabilities from the same point-in-time features the statistical analyzer
uses. Falls back to the statistical analyzer when no usable model exists.
"""

import logging
import os
from typing import List, Optional

import numpy as np

from f1_predictor.analyzer import PredictionAnalyzer
from f1_predictor.features import FEATURE_NAMES, extract_all
from f1_predictor.models import DriverPrediction, RaceContext
from f1_predictor.probability import normalize, scale_to_sum
from f1_predictor.training import DEFAULT_MODEL_PATH, load_bundle, resolve_model_path

logger = logging.getLogger(__name__)


class MLPredictionAnalyzer(PredictionAnalyzer):
    """Analyzer backed by a trained scikit-learn bundle, with statistical fallback."""

    name = "ml"

    def __init__(self, model_path: Optional[str] = None, bundle: Optional[dict] = None):
        super().__init__()
        self.model_path = resolve_model_path(model_path or DEFAULT_MODEL_PATH)
        self.bundle: Optional[dict] = None
        self.win_model = None
        self.podium_model = None
        self.load_error: Optional[str] = None
        if bundle is not None:
            self._adopt_bundle(bundle)
        else:
            self._load_model()

    @property
    def model_loaded(self) -> bool:
        return self.win_model is not None

    def _adopt_bundle(self, bundle: dict) -> None:
        names = bundle.get("feature_names")
        if names != FEATURE_NAMES:
            self.load_error = (
                "Model was trained with a different feature set; retrain with `python train_model.py`."
            )
            logger.warning(self.load_error)
            return
        self.bundle = bundle
        self.win_model = bundle["win_model"]
        self.podium_model = bundle.get("podium_model")

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            self.load_error = f"ML model not found at {self.model_path}. Run `python train_model.py` first."
            logger.warning(self.load_error)
            return
        try:
            bundle = load_bundle(self.model_path)
        except Exception as e:  # corrupt or incompatible pickle
            self.load_error = f"Failed to load ML model: {e}"
            logger.error(self.load_error)
            return
        self._adopt_bundle(bundle)
        if self.model_loaded:
            logger.info("ML model loaded from %s (trained %s)", self.model_path, bundle.get("trained_at", "?"))

    def analyze(self, context: RaceContext, top_n: Optional[int] = None) -> List[DriverPrediction]:
        if not self.model_loaded:
            logger.info("Using statistical analysis (ML model not available)")
            return super().analyze(context, top_n)
        if not context.entries:
            return []

        all_features = extract_all(context)
        X = np.array([f.vector() for f in all_features], dtype=float)

        raw_win = self.win_model.predict_proba(X)[:, 1]
        win_probs = normalize(raw_win)
        if self.podium_model is not None:
            raw_podium = self.podium_model.predict_proba(X)[:, 1]
            podium_probs = scale_to_sum(raw_podium, target=3.0, cap=1.0)
        else:
            from f1_predictor.probability import plackett_luce_top3
            podium_probs = plackett_luce_top3(win_probs)

        predictions: List[DriverPrediction] = []
        for features, p_win, p_podium, raw in zip(all_features, win_probs, podium_probs, raw_win):
            factors = self.score_features(features, context)
            reasoning = self.generate_reasoning(features, context, factors)
            reasoning.insert(0, f"Model win probability: {p_win * 100:.1f}% (raw {raw * 100:.1f}%)")
            predictions.append(
                DriverPrediction(
                    driver=features.entry.driver,
                    constructor=features.entry.constructor,
                    win_probability=p_win,
                    podium_probability=p_podium,
                    score=float(raw) * 100,
                    factors=dict(features.factors),
                    reasoning=reasoning,
                    grid_position=features.qualifying_position,
                )
            )

        predictions.sort(key=lambda p: p.win_probability, reverse=True)
        logger.info("ml: %d predictions for %s", len(predictions), context.race.race_name)
        return predictions[:top_n] if top_n else predictions
