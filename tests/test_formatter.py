import json
from datetime import datetime, timezone

from f1_predictor.analyzer import PredictionAnalyzer
from f1_predictor.formatter import ResultFormatter, result_to_dict
from f1_predictor.models import PredictionResult


def _result(ctx, top=3):
    preds = PredictionAnalyzer().analyze(ctx)
    return PredictionResult(race=ctx.race, predictions=preds[:top], generated_at=datetime.now(timezone.utc),
                            data_sources=["test"], data_completeness=0.8, model_name="statistical",
                            weather=ctx.weather, qualifying_available=ctx.has_qualifying, notes=list(ctx.notes),
                            actual_results=ctx.actual_results)


def test_text_output_mentions_drivers_and_probabilities(sample_context):
    text = ResultFormatter().format_prediction(_result(sample_context))
    assert "TOP 3 PREDICTIONS" in text
    assert "Win " in text and "Podium " in text
    assert sample_context.entries[0].driver.surname in text or "Alpha" in text
    verbose = ResultFormatter().format_prediction(_result(sample_context), verbose=True)
    assert "Factors:" in verbose and "Championship Position" in verbose


def test_actual_result_shown_for_completed_race(store):
    ctx = store.build_context(2025, 2, include_actual=True)
    text = ResultFormatter().format_prediction(_result(ctx))
    assert "Actual result: P1 Alpha-Two" in text


def test_json_round_trip(sample_context):
    payload = result_to_dict(_result(sample_context))
    dumped = json.loads(json.dumps(payload))
    assert dumped["race"]["round"] == 5
    assert len(dumped["predictions"]) == 3
    p = dumped["predictions"][0]
    assert {"driver", "constructor", "win_probability", "podium_probability", "factors", "reasoning", "grid_position"} <= set(p)


def test_comparison_table(sample_context):
    results = {"statistical": _result(sample_context), "ml": _result(sample_context)}
    text = ResultFormatter().format_comparison(results, top_n=2)
    assert "statistical" in text and "ml" in text and "2." in text
