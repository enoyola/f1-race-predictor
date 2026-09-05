"""Helpers for turning scores into probabilities."""

import math
from typing import List, Sequence


def softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    """Softmax with a temperature. Higher temperature flattens the distribution."""
    if not scores:
        return []
    temperature = max(temperature, 1e-6)
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def normalize(values: Sequence[float]) -> List[float]:
    """Scale non-negative values so they sum to 1 (uniform if all zero)."""
    vals = [max(0.0, float(v)) for v in values]
    total = sum(vals)
    if total <= 0:
        return [1.0 / len(vals)] * len(vals) if vals else []
    return [v / total for v in vals]


def scale_to_sum(values: Sequence[float], target: float, cap: float = 1.0) -> List[float]:
    """
    Scale values so they sum to `target`, with no value above `cap`.

    Values that would exceed the cap are pinned to it and the remainder is
    redistributed proportionally among the rest.
    """
    vals = [max(0.0, float(v)) for v in values]
    if not vals:
        return []
    pinned = [False] * len(vals)
    for _ in range(len(vals) + 1):
        free_total = sum(v for v, p in zip(vals, pinned) if not p)
        remaining = target - sum(cap for p in pinned if p)
        if free_total <= 0 or remaining <= 0:
            break
        factor = remaining / free_total
        changed = False
        for i, (v, p) in enumerate(zip(vals, pinned)):
            if p:
                continue
            scaled = v * factor
            if scaled > cap:
                pinned[i] = True
                changed = True
        if not changed:
            return [cap if p else v * factor for v, p in zip(vals, pinned)]
    return [cap if p else v for v, p in zip(vals, pinned)]


def plackett_luce_top3(weights: Sequence[float]) -> List[float]:
    """
    Probability that each competitor finishes in the top three under a
    Plackett-Luce model, given per-competitor strength weights (e.g. softmax
    win probabilities).

    P(i first)  = w_i / S
    P(i second) = sum_j  (w_j / S) * (w_i / (S - w_j))
    P(i third)  = sum_j sum_k (w_j / S) * (w_k / (S - w_j)) * (w_i / (S - w_j - w_k))
    """
    w = [max(1e-12, float(x)) for x in weights]
    n = len(w)
    if n <= 3:
        return [1.0] * n
    S = sum(w)
    result = []
    for i in range(n):
        wi = w[i]
        p1 = wi / S
        p2 = 0.0
        p3 = 0.0
        for j in range(n):
            if j == i:
                continue
            wj = w[j]
            pj = wj / S
            s_minus_j = S - wj
            p2 += pj * (wi / s_minus_j)
            for k in range(n):
                if k == i or k == j:
                    continue
                wk = w[k]
                p3 += pj * (wk / s_minus_j) * (wi / (s_minus_j - wk))
        result.append(min(1.0, p1 + p2 + p3))
    return result
