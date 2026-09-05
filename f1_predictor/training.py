"""
Training-set construction and model training.

Every training row is built from a point-in-time RaceContext, so the features
match exactly what the predictor sees before a real race. Each race is also
added a second time with qualifying hidden, so the model learns to predict
before qualifying has happened.
"""

import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from f1_predictor.features import FEATURE_NAMES, extract_all
from f1_predictor.history import HistoricalDataStore
from f1_predictor.models import RaceContext

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/f1_predictor.joblib"
LEGACY_MODEL_PATH = "models/f1_predictor.pkl"


def resolve_model_path(path: str) -> str:
    """Resolve a model path relative to the CWD, then the repository root."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, path)
    return candidate if os.path.exists(candidate) else path


@dataclass
class TrainingRow:
    season: int
    round: int
    race_name: str
    driver_id: str
    features: List[float]
    won: int
    podium: int
    qualifying_masked: bool


def rows_for_context(context: RaceContext, masked: bool = False) -> List[TrainingRow]:
    """Build labelled rows for one race context (requires actual results)."""
    if not context.actual_results:
        return []
    finish = {r.driver.driver_id: r.position for r in context.actual_results}
    winner = next((d for d, pos in finish.items() if pos == 1), None)
    if winner is None:
        return []
    rows = []
    for f in extract_all(context):
        d = f.entry.driver.driver_id
        if d not in finish:
            continue  # entered but did not start
        rows.append(
            TrainingRow(
                season=context.race.season,
                round=context.race.round,
                race_name=context.race.race_name,
                driver_id=d,
                features=f.vector(),
                won=1 if d == winner else 0,
                podium=1 if finish[d] <= 3 else 0,
                qualifying_masked=masked,
            )
        )
    return rows


def build_training_rows(
    store: HistoricalDataStore,
    seasons: Iterable[int],
    use_weather: bool = True,
    augment_masked: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    only_before: Optional[Tuple[int, int]] = None,
) -> List[TrainingRow]:
    """
    Build rows for every completed race in the given seasons.

    Args:
        only_before: (season, round) - exclude that race and everything after it
    """
    rows: List[TrainingRow] = []
    for season in seasons:
        data = store.load_season(season, with_qualifying=True)
        for rnd in data.completed_rounds:
            if only_before is not None and (season, rnd) >= only_before:
                continue
            try:
                context = store.build_context(season, rnd, include_actual=True, use_weather=use_weather)
            except Exception as e:
                logger.warning(f"Skipping {season} round {rnd}: {e}")
                continue
            race_rows = rows_for_context(context)
            if augment_masked and context.has_qualifying:
                masked = replace(context, qualifying_results=[])
                race_rows += rows_for_context(masked, masked=True)
            rows.extend(race_rows)
            if progress:
                progress(f"{season} R{rnd:02d} {context.race.race_name}: {len(race_rows)} rows")
    return rows


def rows_to_arrays(rows: List[TrainingRow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([r.features for r in rows], dtype=float)
    y_win = np.array([r.won for r in rows], dtype=int)
    y_podium = np.array([r.podium for r in rows], dtype=int)
    return X, y_win, y_podium


DEFAULT_ALGORITHM = "logreg"


def make_classifier(algorithm: str = DEFAULT_ALGORITHM, random_state: int = 42):
    if algorithm == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if algorithm == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1.0,
            random_state=random_state,
        )
    if algorithm == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    raise ValueError(f"Unknown algorithm: {algorithm}")


def train_models(X: np.ndarray, y_win: np.ndarray, y_podium: np.ndarray, algorithm: str = DEFAULT_ALGORITHM, random_state: int = 42):
    win_model = make_classifier(algorithm, random_state).fit(X, y_win)
    podium_model = make_classifier(algorithm, random_state).fit(X, y_podium)
    return win_model, podium_model


def feature_importances(model) -> Optional[Dict[str, float]]:
    """
    Relative importance per feature (sums to 1).

    Tree ensembles expose feature_importances_. For the logistic-regression
    pipeline the features are standardized first, so |coefficient| is a fair
    importance measure.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        coefs = feature_coefficients(model)
        if coefs is None:
            return None
        total = sum(abs(v) for v in coefs.values()) or 1.0
        return {name: abs(v) / total for name, v in coefs.items()}
    return {name: float(v) for name, v in zip(FEATURE_NAMES, importances)}


def feature_coefficients(model) -> Optional[Dict[str, float]]:
    """Signed standardized coefficients for linear models (None otherwise)."""
    estimator = model
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return None
    return {name: float(v) for name, v in zip(FEATURE_NAMES, coef[0])}


def make_bundle(win_model, podium_model, train_seasons: List[int], n_samples: int, metrics: Optional[dict] = None, algorithm: str = DEFAULT_ALGORITHM) -> dict:
    import sklearn
    return {
        "win_model": win_model,
        "podium_model": podium_model,
        "feature_names": list(FEATURE_NAMES),
        "algorithm": algorithm,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seasons": list(train_seasons),
        "n_samples": int(n_samples),
        "metrics": metrics or {},
        "feature_importances": feature_importances(win_model),
        "feature_coefficients": feature_coefficients(win_model),
    }


def save_bundle(bundle: dict, path: str = DEFAULT_MODEL_PATH) -> str:
    import joblib
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str = DEFAULT_MODEL_PATH) -> dict:
    import joblib
    import sklearn
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "win_model" not in bundle:
        raise ValueError("Model file is not a valid bundle")
    trained_with = bundle.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        logger.warning(
            "Model trained with scikit-learn %s but %s is installed; retrain if predictions look off",
            trained_with, sklearn.__version__,
        )
    return bundle
