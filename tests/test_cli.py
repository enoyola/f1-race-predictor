import json

import pytest

from f1_predictor import cli
from f1_predictor.models import PredictionError


def test_parse_arguments_defaults():
    args = cli.parse_arguments([])
    assert args.top == 3 and not args.ml and not args.json


def test_parse_arguments_validation():
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--top", "0"])
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--ml", "--compare"])
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--race", "monza", "--list-races"])


class FakeEngine:
    def __init__(self, store, **kwargs):
        self.store = store
        self.kwargs = kwargs
        from f1_predictor.engine import PredictionEngine
        self._real = PredictionEngine.__new__(PredictionEngine)
        self._real.top_n = kwargs.get("top_n")
        self._real.verbose = kwargs.get("verbose", False)
        self._real.store = store
        self._real.use_weather = False
        from f1_predictor.formatter import ResultFormatter
        from f1_predictor.analyzer import PredictionAnalyzer
        self._real.formatter = ResultFormatter()
        self._real._analyzers = {"statistical": PredictionAnalyzer()}
        self._real.analyzer = self._real._analyzers["statistical"]
        self._real.model_path = None

    def current_season(self):
        return 2025

    def get_schedule(self, season):
        return self.store.schedule(season)

    def completed_rounds(self, season):
        return self.store.completed_rounds(season)

    def get_next_race(self):
        return self.store.schedule(2025)[4]

    def resolve_race(self, selector, season=None):
        return self._real.resolve_race.__func__(self, selector, season)

    def predict_race(self, season, rnd, model=None):
        ctx = self.store.build_context(season, rnd, include_actual=True, use_weather=False)
        return self._real.predict_context(ctx)

    def compare(self, season, rnd, models=None):
        ctx = self.store.build_context(season, rnd, include_actual=True, use_weather=False)
        return {"statistical": self._real.predict_context(ctx), "ml": self._real.predict_context(ctx)}

    def format_result(self, result):
        return self._real.format_result(result)

    def format_comparison(self, results):
        return self._real.format_comparison(results)

    def to_dict(self, result):
        return self._real.to_dict(result)

    def format_error(self, error):
        return self._real.format_error(error)


@pytest.fixture
def patched_cli(monkeypatch, store):
    monkeypatch.setattr(cli, "PredictionEngine", lambda **kw: FakeEngine(store, **kw))
    return cli


def test_main_json_next_race(patched_cli, capsys):
    assert patched_cli.main(["--json", "--top", "2", "--no-weather"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["race"]["round"] == 5 and len(payload["predictions"]) == 2


def test_main_specific_race_by_name_shows_actual(patched_cli, capsys):
    assert patched_cli.main(["--race", "utopia grand prix 2", "--season", "2025"]) == 0
    out = capsys.readouterr().out
    assert "Actual result" in out


def test_main_list_races(patched_cli, capsys):
    assert patched_cli.main(["--list-races"]) == 0
    out = capsys.readouterr().out
    assert "[done]" in out and "[upcoming]" in out and "(sprint)" in out


def test_main_compare(patched_cli, capsys):
    assert patched_cli.main(["--compare", "--top", "2"]) == 0
    assert "Model comparison" in capsys.readouterr().out


def test_main_unknown_race_errors(patched_cli, capsys):
    assert patched_cli.main(["--race", "nowhere"]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_resolve_race_by_round_and_ambiguity(store):
    engine = FakeEngine(store)
    assert engine.resolve_race("3", 2025).round == 3
    with pytest.raises(PredictionError):
        engine.resolve_race("Utopia", 2025)  # matches several
    with pytest.raises(PredictionError):
        engine.resolve_race("99", 2025)
