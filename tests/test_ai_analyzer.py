import json
from types import SimpleNamespace

import pytest

from f1_predictor.ai_analyzer import AIPredictionAnalyzer, OUTPUT_SCHEMA
from f1_predictor.analyzer import PredictionAnalyzer
from f1_predictor.cache import DataCache
from f1_predictor.engine import blend_results
from f1_predictor.models import PredictionResult
from datetime import datetime, timezone


class FakeMessages:
    def __init__(self, answer, stop_reason="end_turn"):
        self.answer = answer
        self.stop_reason = stop_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self.answer) if not isinstance(self.answer, str) else self.answer
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
        )


class FakeClient:
    def __init__(self, answer, stop_reason="end_turn"):
        self.messages = FakeMessages(answer, stop_reason)


def _answer(context, favourite="aaa"):
    preds = []
    for i, e in enumerate(context.entries):
        d = e.driver.driver_id
        preds.append({"driver_id": d, "win_probability": 50 if d == favourite else 10, "podium_probability": 90 if d == favourite else 40, "note": f"note {d}"})
    return {"analysis": "Favourite is Alpha-One. Threats from Beta. Wildcard rain.", "key_factors": ["grid", "form"], "predictions": preds}


def test_ai_analyzer_normalizes_and_annotates(sample_context, tmp_path):
    client = FakeClient(_answer(sample_context, favourite="bba"))
    analyzer = AIPredictionAnalyzer(client=client, cache=DataCache(str(tmp_path)))
    assert analyzer.model_loaded
    preds = analyzer.analyze(sample_context)
    assert preds[0].driver.driver_id == "bba"
    assert abs(sum(p.win_probability for p in preds) - 1) < 1e-9
    assert abs(sum(p.podium_probability for p in preds) - 3) < 1e-6
    assert preds[0].reasoning[0] == "AI analyst: note bba"
    assert analyzer.last_analysis.startswith("Favourite")
    assert analyzer.last_key_factors == ["grid", "form"]
    assert analyzer.last_usage["input_tokens"] == 1200
    call = client.messages.calls[0]
    assert call["output_config"]["format"]["schema"] == OUTPUT_SCHEMA
    assert "Pre-race briefing" in call["messages"][0]["content"]


def test_ai_analyzer_caches_per_race(sample_context, tmp_path):
    client = FakeClient(_answer(sample_context))
    cache = DataCache(str(tmp_path))
    AIPredictionAnalyzer(client=client, cache=cache).analyze(sample_context)
    AIPredictionAnalyzer(client=client, cache=cache).analyze(sample_context)
    assert len(client.messages.calls) == 1


def test_missing_drivers_get_floor(sample_context, tmp_path):
    answer = _answer(sample_context)
    answer["predictions"] = answer["predictions"][:2]
    preds = AIPredictionAnalyzer(client=FakeClient(answer), cache=DataCache(str(tmp_path))).analyze(sample_context)
    assert len(preds) == len(sample_context.entries)
    assert min(p.win_probability for p in preds) > 0


def test_refusal_and_bad_json_fall_back(sample_context, tmp_path):
    analyzer = AIPredictionAnalyzer(client=FakeClient(_answer(sample_context), stop_reason="refusal"), cache=DataCache(str(tmp_path)))
    preds = analyzer.analyze(sample_context)
    assert "declined" in analyzer.load_error
    assert len(preds) == len(sample_context.entries)  # statistical fallback
    analyzer2 = AIPredictionAnalyzer(client=FakeClient("not json"), cache=DataCache(str(tmp_path / "b")))
    analyzer2.analyze(sample_context)
    assert "invalid JSON" in analyzer2.load_error


def test_unavailable_without_credentials(monkeypatch, sample_context):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("f1_predictor.ai_analyzer.credentials_available", lambda: False)
    analyzer = AIPredictionAnalyzer()
    assert not analyzer.model_loaded
    assert "credentials" in analyzer.load_error
    assert len(analyzer.analyze(sample_context)) == len(sample_context.entries)


def test_briefing_includes_reference_picks(sample_context, tmp_path):
    analyzer = AIPredictionAnalyzer(client=FakeClient(_answer(sample_context)), cache=DataCache(str(tmp_path)),
                                    reference_analyzers=[PredictionAnalyzer()])
    briefing = analyzer.build_briefing(sample_context)
    assert "statistical" in briefing["model_picks"]
    assert briefing["race"]["qualifying_available"]
    assert len(briefing["drivers"]) == len(sample_context.entries)
    assert briefing["drivers"][0]["grid"] is not None


def _result(ctx, preds, name, analysis=None):
    return PredictionResult(race=ctx.race, predictions=preds, generated_at=datetime.now(timezone.utc), data_sources=["t"],
                            data_completeness=1.0, model_name=name, qualifying_available=True, analysis=analysis)


def test_blend_results(sample_context, tmp_path):
    stat = PredictionAnalyzer().analyze(sample_context)
    ai = AIPredictionAnalyzer(client=FakeClient(_answer(sample_context, favourite="ggb")), cache=DataCache(str(tmp_path))).analyze(sample_context)
    verdict = blend_results(sample_context, {"statistical": _result(sample_context, stat, "statistical"), "ai": _result(sample_context, ai, "ai", "The analysis")})
    assert verdict.model_name == "verdict"
    assert abs(sum(p.win_probability for p in verdict.predictions) - 1) < 1e-9
    assert verdict.analysis == "The analysis"
    assert set(verdict.components[stat[0].driver.driver_id]) == {"statistical", "ai"}
    top = verdict.predictions[0]
    assert top.reasoning[0].startswith("Blend of:")
    assert any(r.startswith("AI analyst:") for r in top.reasoning)
    assert any("2 model(s)" in n for n in verdict.notes)
