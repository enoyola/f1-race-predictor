from dataclasses import replace

import pytest

from f1_predictor.features import (
    FEATURE_NAMES, avg_finish, championship_score, circuit_score, dnf_rate, extract_all, extract_features,
    form_score, position_score, reliability_score, teammate_delta, teammate_score,
)
from f1_predictor.models import Entry, is_finished
from tests.conftest import DRIVERS, make_race, make_results


def test_is_finished_handles_jolpica_and_ergast_statuses():
    assert is_finished("Finished")
    assert is_finished("Lapped")
    assert is_finished("+1 Lap")
    assert not is_finished("Retired")
    assert not is_finished("Accident")
    assert not is_finished("")


def test_position_score_curve():
    assert position_score(1) == 100
    assert position_score(2) == 90
    assert position_score(3) == 80
    assert position_score(5) == 70
    assert position_score(10) == 50
    assert position_score(15) == 25
    assert position_score(20) == 0
    assert position_score(None) == 50
    assert position_score(0) == 50


def test_form_score_rewards_wins_and_penalizes_retirements():
    race1 = make_race(2025, 1)
    race2 = make_race(2025, 2)
    winner_results = make_results(race1, [0, 1, 2, 3, 4, 5]) + make_results(race2, [0, 1, 2, 3, 4, 5])
    assert form_score("aaa", winner_results) == 100.0
    lapped = make_results(race1, [0, 1, 2, 3, 4, 5], statuses={5: "Lapped"})
    retired = make_results(race1, [0, 1, 2, 3, 4, 5], statuses={5: "Retired"})
    assert form_score("ggb", lapped) > form_score("ggb", retired)
    assert form_score("nobody", winner_results) == 50.0


def test_form_uses_most_recent_races_only():
    races = [make_race(2025, r) for r in range(1, 8)]
    # driver 0 wins the first two races then finishes last five times
    results = []
    for r in races[:2]:
        results += make_results(r, [0, 1, 2, 3, 4, 5])
    for r in races[2:]:
        results += make_results(r, [1, 2, 3, 4, 5, 0])
    last_five_only = [r for r in results if r.race.round > 2]
    assert form_score("aaa", results) == form_score("aaa", last_five_only)  # early wins fell out of the window
    assert form_score("aaa", results) < form_score("aab", results)


def test_championship_score_leader_is_100(sample_context):
    leader = sample_context.driver_standings[0]
    assert championship_score(leader.driver.driver_id, sample_context.driver_standings) == 100.0
    assert championship_score("nobody", sample_context.driver_standings) == 50.0
    assert championship_score("aaa", []) == 50.0


def test_reliability_and_dnf_rate():
    race = make_race(2025, 1)
    results = make_results(race, [0, 1, 2, 3, 4, 5], statuses={0: "Accident"})
    assert dnf_rate("aaa", results) == 1.0
    assert reliability_score("aaa", results) == 0.0
    assert reliability_score("aab", results) == 100.0
    assert reliability_score("nobody", results) == 85.0


def test_teammate_delta_sign():
    race = make_race(2025, 1)
    results = make_results(race, [0, 1, 2, 3, 4, 5])  # aaa P1, aab P2 (same team)
    assert teammate_delta("aaa", "alpha", results) == 1.0
    assert teammate_delta("aab", "alpha", results) == -1.0
    assert teammate_score("aaa", "alpha", results) == 55.0
    assert teammate_delta("aaa", "alpha", []) is None


def test_circuit_score():
    race = make_race(2024, 1)
    history = make_results(race, [0, 1, 2, 3, 4, 5])
    assert circuit_score("aaa", history) == 100.0   # win = 30/30
    assert circuit_score("aab", history) == 50.0    # podium = 15/30
    assert circuit_score("ggb", history) == pytest.approx(100 * 5 / 30)  # points finish = 5/30
    assert circuit_score("nobody", history) == 50.0


def test_avg_finish_default():
    assert avg_finish("nobody", []) == 12.0


def test_extract_features_vector_matches_feature_names(sample_context):
    feats = extract_all(sample_context)
    assert len(feats) == len(sample_context.entries)
    for f in feats:
        assert len(f.vector()) == len(FEATURE_NAMES)
        assert set(f.factors) <= set(FEATURE_NAMES)
        assert f.values["has_qualifying"] == 1.0
        assert 0 <= f.values["season_progress"] <= 1


def test_extract_features_without_qualifying(sample_context):
    masked = replace(sample_context, qualifying_results=[])
    entry = masked.entries[0]
    f = extract_features(masked, entry)
    assert f.values["has_qualifying"] == 0.0
    assert f.values["qualifying"] == 50.0
    assert f.values["grid_position"] == 21.0
    assert f.qualifying_position is None


def test_wet_flag(wet_context):
    f = extract_features(wet_context, wet_context.entries[0])
    assert f.values["wet"] == 1.0
    assert f.values["weather_known"] == 1.0
