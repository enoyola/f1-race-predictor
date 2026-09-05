import math

import pytest

from f1_predictor.probability import normalize, plackett_luce_top3, scale_to_sum, softmax


def test_softmax_sums_to_one_and_keeps_order():
    probs = softmax([70, 60, 50], temperature=5)
    assert math.isclose(sum(probs), 1.0)
    assert probs[0] > probs[1] > probs[2]


def test_softmax_temperature_flattens():
    sharp = softmax([70, 50], temperature=2)
    flat = softmax([70, 50], temperature=50)
    assert sharp[0] > flat[0] > 0.5


def test_softmax_empty():
    assert softmax([], 5) == []


def test_normalize_uniform_when_all_zero():
    assert normalize([0, 0, 0, 0]) == [0.25] * 4


def test_normalize_scales():
    assert normalize([1, 3]) == [0.25, 0.75]


def test_plackett_luce_top3_sums_to_three():
    probs = plackett_luce_top3([0.4, 0.3, 0.2, 0.05, 0.05])
    assert math.isclose(sum(probs), 3.0, abs_tol=1e-9)
    assert probs[0] > probs[1] > probs[2] > probs[3]
    assert all(0 <= p <= 1 for p in probs)


def test_plackett_luce_small_field():
    assert plackett_luce_top3([0.5, 0.5]) == [1.0, 1.0]


def test_scale_to_sum_caps_and_redistributes():
    scaled = scale_to_sum([0.9, 0.8, 0.7, 0.1, 0.05, 0.05], target=3.0, cap=1.0)
    assert math.isclose(sum(scaled), 3.0, abs_tol=1e-6)
    assert max(scaled) <= 1.0 + 1e-9
    assert scaled[0] == pytest.approx(1.0)


def test_scale_to_sum_plain():
    assert scale_to_sum([1, 1], target=1.0) == [0.5, 0.5]
