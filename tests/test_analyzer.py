import math
from dataclasses import replace

from f1_predictor.analyzer import PredictionAnalyzer


def test_analyze_returns_normalized_probabilities(sample_context):
    preds = PredictionAnalyzer().analyze(sample_context)
    assert len(preds) == len(sample_context.entries)
    assert math.isclose(sum(p.win_probability for p in preds), 1.0)
    assert math.isclose(sum(p.podium_probability for p in preds), 3.0, abs_tol=1e-6)
    assert preds == sorted(preds, key=lambda p: p.win_probability, reverse=True)
    assert preds[0].grid_position is not None
    assert preds[0].reasoning
    assert preds[0].confidence == preds[0].win_probability * 100


def test_top_n(sample_context):
    assert len(PredictionAnalyzer().analyze(sample_context, top_n=2)) == 2


def test_sharper_with_lower_temperature(sample_context):
    sharp = PredictionAnalyzer(temperature=2).analyze(sample_context)
    flat = PredictionAnalyzer(temperature=30).analyze(sample_context)
    assert sharp[0].win_probability > flat[0].win_probability


def test_wet_weights_reduce_qualifying(sample_context, wet_context):
    analyzer = PredictionAnalyzer()
    assert analyzer.active_weights(wet_context)["qualifying"] < analyzer.active_weights(sample_context)["qualifying"]
    preds = analyzer.analyze(wet_context)
    assert any("wet" in r.lower() for r in preds[0].reasoning)


def test_missing_qualifying_drops_factor_or_uses_sprint(store):
    analyzer = PredictionAnalyzer()
    ctx = store.build_context(2025, 6)  # no qualifying, no sprint
    from f1_predictor.features import extract_all
    factors = analyzer.score_features(extract_all(ctx)[0], ctx)
    assert "qualifying" not in factors and "sprint" not in factors

    sprint_ctx = replace(store.build_context(2025, 3), qualifying_results=[])
    f = extract_all(sprint_ctx)[0]
    factors = analyzer.score_features(f, sprint_ctx)
    assert factors["qualifying"] == f.values["sprint"]


def test_combine_factors_renormalizes_weights():
    analyzer = PredictionAnalyzer()
    assert analyzer.combine_factors({"championship": 80, "form": 80}) == 80
    assert analyzer.combine_factors({}) == 50
    assert analyzer.combine_factors({"unknown": 10}) == 50


def test_empty_context_returns_nothing(sample_context):
    assert PredictionAnalyzer().analyze(replace(sample_context, entries=[])) == []


def test_score_context_matches_entries(sample_context):
    scores = PredictionAnalyzer().score_context(sample_context)
    assert len(scores) == len(sample_context.entries)
    assert all(0 <= s <= 100 for s in scores)
