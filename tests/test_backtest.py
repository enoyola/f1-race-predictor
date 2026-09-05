import math

from f1_predictor.analyzer import PredictionAnalyzer
from f1_predictor.backtest import (
    BASELINE_LEADER, BASELINE_POLE, ModelMetrics, baseline_predictions, calibrate_temperature, format_report,
    run_backtest, score_predictions,
)
from f1_predictor.models import DriverPrediction
from tests.conftest import DRIVERS, make_race, make_results


def _pred(idx, win, podium):
    d, c = DRIVERS[idx]
    return DriverPrediction(driver=d, constructor=c, win_probability=win, podium_probability=podium, score=win * 100, factors={}, reasoning=[])


def test_score_predictions_hits():
    actual = make_results(make_race(2025, 1), [0, 1, 2, 3, 4, 5])
    preds = [_pred(0, 0.5, 0.9), _pred(1, 0.3, 0.8), _pred(2, 0.1, 0.6), _pred(3, 0.1, 0.4)]
    m = score_predictions(preds, actual)
    assert m.winner_hits == 1 and m.winner_in_top3 == 1 and m.podium_hits == 3
    assert math.isclose(m.log_loss_sum, -math.log(0.5))
    assert m.scored_races == 1
    d = m.to_dict()
    assert d["top1_accuracy"] == 1.0 and d["podium_precision"] == 1.0


def test_score_predictions_miss():
    actual = make_results(make_race(2025, 1), [3, 4, 5, 0, 1, 2])
    preds = [_pred(0, 0.5, 0.9), _pred(1, 0.3, 0.8), _pred(2, 0.2, 0.6)]
    m = score_predictions(preds, actual)
    assert m.winner_hits == 0 and m.winner_in_top3 == 0 and m.podium_hits == 0
    assert m.scored_races == 0  # winner not among predicted drivers
    assert m.to_dict()["log_loss"] is None


def test_metrics_add():
    a, b = ModelMetrics(races=1, winner_hits=1), ModelMetrics(races=1, winner_hits=0)
    a.add(b)
    assert a.races == 2 and a.to_dict()["top1_accuracy"] == 0.5


def test_baselines(sample_context):
    pole = baseline_predictions(sample_context, BASELINE_POLE)
    assert pole[0].grid_position == 1 and pole[0].win_probability == 1.0
    leader = baseline_predictions(sample_context, BASELINE_LEADER)
    assert leader[0].driver.driver_id == sample_context.driver_standings[0].driver.driver_id


def test_calibrate_temperature_prefers_sharper_when_scores_are_right():
    scored = [([90, 50, 40], ["a", "b", "c"], "a")] * 5
    result = calibrate_temperature(scored, candidates=[2, 10, 50])
    assert result["best_temperature"] == 2
    assert calibrate_temperature([]) == {}


def test_run_backtest_end_to_end(store):
    report = run_backtest(store, [2025], include_ml=False, use_weather=False)
    assert report.seasons == [2025]
    assert len(report.races) == 4
    stat = report.summary["statistical"]
    assert stat["2025"]["races"] == 4 and stat["overall"]["races"] == 4
    assert report.summary["pole"]["overall"]["races"] == 4
    assert report.calibration["best_temperature"]
    text = format_report(report)
    assert "statistical" in text and "pole" in text
    d = report.to_dict()
    assert d["races"][0]["predictions"]["statistical"]["top3"]
