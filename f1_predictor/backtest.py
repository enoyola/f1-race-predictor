"""
Walk-forward backtesting.

For every race in the test seasons a point-in-time context is built, each
model predicts from it, and the prediction is scored against the real result.
The ML model is retrained for each test season using only earlier seasons.
Two naive baselines (pole sitter wins, championship leader wins) show whether
the models add anything over the obvious picks.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

from f1_predictor.analyzer import PredictionAnalyzer
from f1_predictor.history import HistoricalDataStore
from f1_predictor.models import DriverPrediction, RaceContext, RaceResult
from f1_predictor.probability import softmax
from f1_predictor.training import build_training_rows, rows_to_arrays, train_models, TrainingRow

logger = logging.getLogger(__name__)

BASELINE_POLE = "pole"
BASELINE_LEADER = "leader"
MODEL_STATISTICAL = "statistical"
MODEL_ML = "ml"


@dataclass
class ModelMetrics:
    races: int = 0
    winner_hits: int = 0
    winner_in_top3: int = 0
    podium_hits: int = 0        # predicted top-3 drivers who really finished top 3
    log_loss_sum: float = 0.0
    brier_sum: float = 0.0
    scored_races: int = 0       # races with probabilities available

    def add(self, other: "ModelMetrics") -> None:
        self.races += other.races
        self.winner_hits += other.winner_hits
        self.winner_in_top3 += other.winner_in_top3
        self.podium_hits += other.podium_hits
        self.log_loss_sum += other.log_loss_sum
        self.brier_sum += other.brier_sum
        self.scored_races += other.scored_races

    def to_dict(self) -> dict:
        r = max(1, self.races)
        s = max(1, self.scored_races)
        return {
            "races": self.races,
            "top1_accuracy": self.winner_hits / r,
            "winner_in_top3": self.winner_in_top3 / r,
            "podium_precision": self.podium_hits / (3 * r),
            "log_loss": (self.log_loss_sum / s) if self.scored_races else None,
            "brier": (self.brier_sum / s) if self.scored_races else None,
        }


def _actual_top(actual: List[RaceResult], n: int) -> List[str]:
    return [r.driver.driver_id for r in sorted(actual, key=lambda r: r.position)[:n]]


def score_predictions(predictions: Sequence[DriverPrediction], actual: List[RaceResult]) -> ModelMetrics:
    """Score a ranked list of predictions against actual results."""
    m = ModelMetrics(races=1)
    if not predictions or not actual:
        return m
    winner = _actual_top(actual, 1)[0]
    podium = set(_actual_top(actual, 3))
    ranked = sorted(predictions, key=lambda p: p.win_probability, reverse=True)
    m.winner_hits = 1 if ranked[0].driver.driver_id == winner else 0
    m.winner_in_top3 = 1 if winner in {p.driver.driver_id for p in ranked[:3]} else 0
    by_podium = sorted(predictions, key=lambda p: p.podium_probability, reverse=True)[:3]
    m.podium_hits = sum(1 for p in by_podium if p.driver.driver_id in podium)

    probs = {p.driver.driver_id: p.win_probability for p in predictions}
    if winner in probs:
        p_win = max(1e-6, probs[winner])
        m.log_loss_sum = -math.log(p_win)
        m.brier_sum = sum((p - (1.0 if d == winner else 0.0)) ** 2 for d, p in probs.items())
        m.scored_races = 1
    return m


def baseline_predictions(context: RaceContext, kind: str) -> List[DriverPrediction]:
    """Deterministic baseline rankings expressed as DriverPrediction objects."""
    entries = context.entries
    if kind == BASELINE_POLE:
        order = sorted(entries, key=lambda e: context.qualifying_position(e.driver.driver_id) or 99)
        if not context.has_qualifying:
            return []
    elif kind == BASELINE_LEADER:
        pos = {s.driver.driver_id: s.position for s in context.driver_standings}
        order = sorted(entries, key=lambda e: pos.get(e.driver.driver_id, 99))
        if not pos:
            return []
    else:
        raise ValueError(kind)
    n = len(order)
    preds = []
    for i, e in enumerate(order):
        rank_prob = 1.0 if i == 0 else 0.0
        preds.append(DriverPrediction(
            driver=e.driver, constructor=e.constructor,
            win_probability=rank_prob, podium_probability=1.0 if i < 3 else 0.0,
            score=float(n - i), factors={}, reasoning=[],
            grid_position=context.qualifying_position(e.driver.driver_id),
        ))
    return preds


@dataclass
class BacktestReport:
    seasons: List[int]
    models: List[str]
    summary: Dict[str, Dict[str, dict]] = field(default_factory=dict)   # model -> season/overall -> metrics
    races: List[dict] = field(default_factory=list)
    generated_at: str = ""
    algorithm: str = "logreg"
    calibration: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "seasons": self.seasons,
            "models": self.models,
            "algorithm": self.algorithm,
            "summary": self.summary,
            "races": self.races,
            "calibration": self.calibration,
        }


def _top3_payload(predictions: Sequence[DriverPrediction]) -> List[dict]:
    ranked = sorted(predictions, key=lambda p: p.win_probability, reverse=True)[:3]
    return [
        {
            "driver_id": p.driver.driver_id,
            "code": p.driver.code,
            "driver": p.driver.full_name,
            "constructor": p.constructor.name,
            "win_probability": round(p.win_probability, 4),
            "podium_probability": round(p.podium_probability, 4),
        }
        for p in ranked
    ]


def run_backtest(
    store: HistoricalDataStore,
    test_seasons: Sequence[int],
    include_ml: bool = True,
    algorithm: str = "logreg",
    use_weather: bool = True,
    min_train_season: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
    temperature: Optional[float] = None,
    extra_analyzers: Optional[Dict[str, PredictionAnalyzer]] = None,
) -> BacktestReport:
    """Walk-forward backtest over the given seasons (extra_analyzers are scored as-is, e.g. the AI analyst)."""
    test_seasons = sorted(test_seasons)
    extra_analyzers = extra_analyzers or {}
    models = [BASELINE_POLE, BASELINE_LEADER, MODEL_STATISTICAL] + ([MODEL_ML] if include_ml else []) + list(extra_analyzers)
    report = BacktestReport(seasons=list(test_seasons), models=models, algorithm=algorithm,
                            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    statistical = PredictionAnalyzer(temperature=temperature)

    # Training rows for every season from min_train_season up to the last test season
    rows: List[TrainingRow] = []
    if include_ml:
        first_train = min_train_season or (test_seasons[0] - 5)
        train_span = list(range(first_train, test_seasons[-1] + 1))
        if progress:
            progress(f"Building training rows for {train_span[0]}-{train_span[-1]}")
        rows = build_training_rows(store, train_span, use_weather=use_weather, progress=None)

    totals: Dict[str, ModelMetrics] = {m: ModelMetrics() for m in models}
    stat_scores_by_race: List[tuple] = []  # (scores, entries ids, winner id) for calibration

    for season in test_seasons:
        per_season: Dict[str, ModelMetrics] = {m: ModelMetrics() for m in models}
        ml_analyzer = None
        if include_ml:
            train_rows = [r for r in rows if r.season < season]
            if len(train_rows) >= 200:
                X, y_win, y_podium = rows_to_arrays(train_rows)
                win_model, podium_model = train_models(X, y_win, y_podium, algorithm=algorithm)
                from f1_predictor.ml_analyzer import MLPredictionAnalyzer
                from f1_predictor.training import make_bundle
                bundle = make_bundle(win_model, podium_model, sorted({r.season for r in train_rows}), len(train_rows), algorithm=algorithm)
                ml_analyzer = MLPredictionAnalyzer(bundle=bundle)
                if progress:
                    progress(f"{season}: ML model trained on {len(train_rows)} rows from {bundle['train_seasons']}")
            elif progress:
                progress(f"{season}: not enough earlier data to train ML ({len(train_rows)} rows), skipping ML")

        data = store.load_season(season, with_qualifying=True)
        for rnd in data.completed_rounds:
            try:
                context = store.build_context(season, rnd, include_actual=True, use_weather=use_weather)
            except Exception as e:
                logger.warning(f"Skipping {season} round {rnd}: {e}")
                continue
            actual = context.actual_results or []
            if not actual or not context.entries:
                continue

            race_record = {
                "season": season,
                "round": rnd,
                "race_name": context.race.race_name,
                "date": context.race.date.date().isoformat(),
                "is_sprint": context.race.is_sprint,
                "wet": bool(context.weather and context.weather.is_wet),
                "actual_winner": _actual_top(actual, 1)[0],
                "actual_podium": _actual_top(actual, 3),
                "pole": next((q.driver.driver_id for q in context.qualifying_results if q.position == 1), None),
                "predictions": {},
            }

            preds_by_model: Dict[str, List[DriverPrediction]] = {
                BASELINE_POLE: baseline_predictions(context, BASELINE_POLE),
                BASELINE_LEADER: baseline_predictions(context, BASELINE_LEADER),
                MODEL_STATISTICAL: statistical.analyze(context),
            }
            if include_ml and ml_analyzer is not None:
                preds_by_model[MODEL_ML] = ml_analyzer.analyze(context)
            for extra_name, extra in extra_analyzers.items():
                try:
                    preds_by_model[extra_name] = extra.analyze(context)
                except Exception as e:
                    logger.warning("%s failed for %s: %s", extra_name, context.race.race_name, e)

            scores = statistical.score_context(context)
            stat_scores_by_race.append((scores, [e.driver.driver_id for e in context.entries], race_record["actual_winner"]))

            for model, preds in preds_by_model.items():
                if not preds:
                    continue
                metrics = score_predictions(preds, actual)
                per_season[model].add(metrics)
                race_record["predictions"][model] = {
                    "top3": _top3_payload(preds),
                    "winner_hit": bool(metrics.winner_hits),
                    "winner_in_top3": bool(metrics.winner_in_top3),
                    "podium_hits": metrics.podium_hits,
                    "log_loss": round(metrics.log_loss_sum, 4) if metrics.scored_races else None,
                }
            report.races.append(race_record)
            if progress:
                stat = race_record["predictions"].get(MODEL_STATISTICAL, {})
                progress(f"{season} R{rnd:02d} {context.race.race_name:<28} winner {race_record['actual_winner']:<14} "
                         f"stat {'✓' if stat.get('winner_hit') else '✗'}"
                         + (f"  ml {'✓' if race_record['predictions'].get(MODEL_ML, {}).get('winner_hit') else '✗'}" if MODEL_ML in race_record["predictions"] else ""))

        for model in models:
            report.summary.setdefault(model, {})[str(season)] = per_season[model].to_dict()
            totals[model].add(per_season[model])

    for model in models:
        report.summary.setdefault(model, {})["overall"] = totals[model].to_dict()

    report.calibration = calibrate_temperature(stat_scores_by_race)
    return report


def calibrate_temperature(scores_by_race: List[tuple], candidates: Optional[Sequence[float]] = None) -> dict:
    """Find the softmax temperature that minimizes winner log loss for the statistical model."""
    if not scores_by_race:
        return {}
    candidates = candidates or [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    results = {}
    for t in candidates:
        total, n = 0.0, 0
        for scores, ids, winner in scores_by_race:
            if winner not in ids:
                continue
            probs = softmax(scores, temperature=t)
            p = max(1e-6, probs[ids.index(winner)])
            total += -math.log(p)
            n += 1
        results[t] = total / max(1, n)
    best = min(results, key=results.get)
    return {"best_temperature": best, "log_loss_by_temperature": {str(k): round(v, 4) for k, v in results.items()}}


def format_report(report: BacktestReport) -> str:
    """Plain-text summary table."""
    lines = []
    lines.append(f"Backtest {report.seasons[0]}-{report.seasons[-1]}  (ML algorithm: {report.algorithm})")
    lines.append("═" * 96)
    header = f"{'model':<12}{'season':<9}{'races':>6}{'top-1':>9}{'in top3':>10}{'podium':>9}{'logloss':>10}{'brier':>8}"
    lines.append(header)
    lines.append("─" * 96)
    for model in report.models:
        for key, m in report.summary.get(model, {}).items():
            ll = f"{m['log_loss']:.3f}" if m["log_loss"] is not None else "  -"
            br = f"{m['brier']:.3f}" if m["brier"] is not None else "  -"
            lines.append(
                f"{model:<12}{key:<9}{m['races']:>6}{m['top1_accuracy']*100:>8.1f}%{m['winner_in_top3']*100:>9.1f}%"
                f"{m['podium_precision']*100:>8.1f}%{ll:>10}{br:>8}"
            )
        lines.append("─" * 96)
    lines.append("top-1: predicted winner was right | in top3: real winner among predicted top 3 | podium: predicted podium drivers who finished top 3")
    lines.append("logloss/brier: lower is better (winner probability quality); baselines have no probabilities")
    if report.calibration:
        lines.append(f"Statistical softmax temperature with lowest log loss: {report.calibration['best_temperature']}")
    return "\n".join(lines)
