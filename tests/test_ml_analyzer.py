import math

import numpy as np
import pytest

from f1_predictor.ml_analyzer import MLPredictionAnalyzer
from f1_predictor.training import (
    build_training_rows, make_bundle, rows_for_context, rows_to_arrays, save_bundle, load_bundle, train_models,
)


@pytest.fixture(scope="module")
def trained_bundle():
    from tests.conftest import FakeFetcher
    from f1_predictor.history import HistoricalDataStore
    f = FakeFetcher()
    f.add_season(2023, rounds=6, completed=6)
    f.add_season(2024, rounds=6, completed=6, winners={5: [2, 0, 1, 3, 4, 5], 6: [2, 1, 0, 3, 4, 5]})
    store = HistoricalDataStore(f, None, history_years=2, min_season=2023)
    rows = build_training_rows(store, [2023, 2024], use_weather=False)
    X, y_win, y_podium = rows_to_arrays(rows)
    win_model, podium_model = train_models(X, y_win, y_podium, algorithm="logreg")
    return make_bundle(win_model, podium_model, [2023, 2024], len(rows), algorithm="logreg"), rows


def test_training_rows_have_labels_and_masked_copies(trained_bundle):
    _, rows = trained_bundle
    races = {(r.season, r.round) for r in rows}
    assert len(races) == 12
    assert sum(r.won for r in rows) == 24            # one winner per race, twice (masked copy)
    assert sum(r.qualifying_masked for r in rows) == len(rows) / 2
    masked = [r for r in rows if r.qualifying_masked][0]
    from f1_predictor.features import FEATURE_NAMES
    assert masked.features[FEATURE_NAMES.index("has_qualifying")] == 0.0


def test_rows_for_context_without_results(sample_context):
    assert rows_for_context(sample_context) == []


def test_ml_analyzer_predicts_normalized(trained_bundle, sample_context):
    bundle, _ = trained_bundle
    analyzer = MLPredictionAnalyzer(bundle=bundle)
    assert analyzer.model_loaded
    preds = analyzer.analyze(sample_context)
    assert len(preds) == len(sample_context.entries)
    assert math.isclose(sum(p.win_probability for p in preds), 1.0)
    assert math.isclose(sum(p.podium_probability for p in preds), 3.0, abs_tol=1e-6)
    assert preds[0].reasoning[0].startswith("Model win probability")


def test_feature_mismatch_falls_back(trained_bundle, sample_context):
    bundle, _ = trained_bundle
    bad = dict(bundle, feature_names=["something_else"])
    analyzer = MLPredictionAnalyzer(bundle=bad)
    assert not analyzer.model_loaded
    assert "retrain" in analyzer.load_error
    preds = analyzer.analyze(sample_context)  # statistical fallback still works
    assert len(preds) == len(sample_context.entries)


def test_missing_model_file(tmp_path, sample_context):
    analyzer = MLPredictionAnalyzer(model_path=str(tmp_path / "nope.joblib"))
    assert not analyzer.model_loaded
    assert analyzer.analyze(sample_context)


def test_bundle_round_trip(tmp_path, trained_bundle):
    bundle, _ = trained_bundle
    path = save_bundle(bundle, str(tmp_path / "m" / "model.joblib"))
    loaded = load_bundle(path)
    assert loaded["feature_names"] == bundle["feature_names"]
    assert loaded["algorithm"] == "logreg"
    assert "sklearn_version" in loaded
