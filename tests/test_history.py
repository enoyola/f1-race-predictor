from f1_predictor.history import compute_constructor_standings, compute_driver_standings
from tests.conftest import make_race, make_results, make_sprint


def test_driver_standings_include_sprint_points_and_exclude_current_round():
    r1, r2, r3 = make_race(2025, 1), make_race(2025, 2, is_sprint=True), make_race(2025, 3)
    results = {1: make_results(r1, [0, 1, 2, 3, 4, 5]), 2: make_results(r2, [1, 0, 2, 3, 4, 5]), 3: make_results(r3, [2, 0, 1, 3, 4, 5])}
    sprints = {2: make_sprint(r2, [1, 0, 2, 3, 4, 5])}

    before3 = compute_driver_standings(results, sprints, before_round=3)
    by_id = {s.driver.driver_id: s for s in before3}
    assert by_id["aaa"].points == 25 + 18 + 7   # win, P2, sprint P2
    assert by_id["aab"].points == 18 + 25 + 8
    assert before3[0].driver.driver_id == "aab"
    assert before3[0].position == 1 and before3[1].position == 2
    assert by_id["aaa"].wins == 1

    full = compute_driver_standings(results, sprints)
    assert {s.driver.driver_id: s.points for s in full}["bba"] == 15 + 15 + 25 + 6  # two P3s, a win, sprint P3


def test_constructor_standings_sum_both_cars():
    r1 = make_race(2025, 1)
    standings = compute_constructor_standings({1: make_results(r1, [0, 1, 2, 3, 4, 5])}, {})
    assert standings[0].constructor.constructor_id == "alpha"
    assert standings[0].points == 25 + 18
    assert standings[0].wins == 1


def test_build_context_is_point_in_time(store):
    ctx = store.build_context(2025, 3)
    assert ctx.rounds_completed == 2
    # standings before round 3 only count rounds 1-2
    total_points = sum(s.points for s in ctx.driver_standings)
    assert total_points == 2 * (25 + 18 + 15 + 12 + 10 + 8)
    # recent results contain the previous season and rounds 1-2 only
    assert {(r.race.season, r.race.round) for r in ctx.recent_results if r.race.season == 2025} == {(2025, 1), (2025, 2)}
    assert all(r.race.date < ctx.race.date for r in ctx.recent_results)
    assert ctx.race.is_sprint
    assert ctx.sprint_results, "sprint results for a sprint weekend should be attached"
    assert ctx.has_qualifying
    assert not ctx.standings_from_previous_season


def test_build_context_round_one_uses_previous_season(store):
    ctx = store.build_context(2025, 1)
    assert ctx.standings_from_previous_season
    assert ctx.rounds_completed == 0
    assert ctx.driver_standings[0].driver.driver_id == "aaa"  # 2024 champion (won 4 of 6)
    assert any("Season opener" in n for n in ctx.notes)


def test_circuit_history_excludes_the_race_itself_and_the_future(store):
    ctx = store.build_context(2025, 4)  # circuit c_one, also used in 2024 r1, r4 and 2025 r1
    seasons_rounds = {(r.race.season, r.race.round) for r in ctx.circuit_history}
    assert (2025, 4) not in seasons_rounds
    assert (2025, 1) in seasons_rounds
    assert (2024, 1) in seasons_rounds and (2024, 4) in seasons_rounds
    assert all(r.race.circuit.circuit_id == "c_one" for r in ctx.circuit_history)


def test_entries_fall_back_to_previous_round_without_qualifying(store):
    ctx = store.build_context(2025, 6)  # no qualifying, no results
    assert not ctx.has_qualifying
    assert len(ctx.entries) == 6
    assert any("previous round" in n for n in ctx.notes)
    assert any("Qualifying not yet available" in n for n in ctx.notes)


def test_include_actual_attaches_results(store):
    ctx = store.build_context(2025, 2, include_actual=True)
    assert ctx.actual_results and ctx.actual_results[0].driver.driver_id == "aab"
    ctx2 = store.build_context(2025, 2, include_actual=False)
    assert ctx2.actual_results is None


def test_unknown_round_raises(store):
    import pytest
    with pytest.raises(ValueError):
        store.build_context(2025, 99)


def test_seasons_are_loaded_once(store, fake_fetcher):
    store.build_context(2025, 3)
    store.build_context(2025, 4)
    assert fake_fetcher.calls.count("results_2025") == 1
    assert fake_fetcher.calls.count("qualifying_2025") == 1
