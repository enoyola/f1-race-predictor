# 🤖 Statistical vs Machine Learning vs AI

The predictor ships three models that read the same pre-race context, plus a verdict that blends them.
This page shows how they differ, how they score on real races, and which one to use.

## Quick comparison

| Aspect | Statistical | Machine learning |
|--------|-------------|------------------|
| Method | Weighted average of 0-100 factor scores, softmax to probabilities | Logistic regression (win and podium classifiers) on 17 features |
| Weights | Hand-set, tuned on the 2023-2026 backtest | Learned from every race since 2020 |
| Training | None | `python train_model.py` (seconds once data is cached) |
| Winner picked (2023-2026 backtest) | **61.0%** | 51.2% |
| Winner in predicted top 3 | **91.5%** | 86.6% |
| Podium precision | 66.3% | **67.5%** |
| Log loss (lower is better) | **1.086** | 1.240 |
| Explainability | Every factor and weight is visible | Standardized coefficients and importances in the bundle |
| Before qualifying | Weights renormalize without the grid | Trained on qualifying-masked copies of every race |

## Backtest, season by season

Walk-forward: each race uses only pre-race data, and the ML model for season N is trained on seasons before N.
Two naive baselines show what the models must beat. Reproduce with `python backtest.py --seasons 2023-2026`.

| Season | Races | Pole sitter | Points leader | Statistical | ML |
|-------:|------:|------------:|--------------:|------------:|---:|
| 2023 | 22 | 68.2% | 86.4% | 81.8% | 86.4% |
| 2024 | 24 | 50.0% | 37.5% | 45.8% | 37.5% |
| 2025 | 24 | 66.7% | 20.8% | 58.3% | 41.7% |
| 2026 (to round 12) | 12 | 75.0% | 33.3% | 58.3% | 33.3% |
| **Overall** | 82 | 63.4% | 45.1% | 61.0% | 51.2% |

Numbers are the share of races where the model's top pick won.

## What the ML model learned

Feature importances of the shipped win model (share of total standardized coefficient magnitude):

```
qualifying          31.0%
form                14.8%
has_qualifying      13.3%
team_points_share   12.9%
grid_position        8.2%
points_share         6.1%
teammate             5.0%
sprint               2.8%
is_sprint_weekend    2.0%
avg_finish_last5     1.3%
championship         0.9%
team                 0.8%
season_progress      0.6%
reliability          0.2%
circuit              0.1%
wet                  0.1%
```

The grid dominates, then form and team strength. Circuit history and weather add almost nothing once
the grid is known, which matches the statistical model's low weights for them.

## Why the old numbers were wrong

Earlier versions of this project reported 99.7% accuracy for the ML model. That figure came from a training
set where the "championship" feature was derived from the finishing position of the same race, which is
the label. Two other features were constants. At prediction time the model was fed real standings it had
never seen in training. The current training set builds every row from a point-in-time context, so the
backtest numbers above are honest out-of-sample results, and they are much lower.

## The AI analyst and the verdict

`--ai` sends the same pre-race briefing to Claude and gets back probabilities plus a written analysis;
`--verdict` averages every available model. The AI analyst is not part of the walk-forward backtest above
because each race costs an API call; run `python backtest.py --ai --seasons 2026` to score it on the current
season once you have a key. Treat its probabilities as a qualitative second opinion rather than a calibrated
model, and read the analysis for the reasoning the numbers cannot show.

## Which one should I use?

- **Statistical** (default): best winner accuracy and best probabilities in the backtest, fully explainable,
  no training step.
- **ML**: slightly better at picking podium finishers and useful as a second opinion. `--compare` shows both.
- **Pole sitter**: if all you want is a single name, the pole sitter is right 63% of the time. The models are
  worth using for the probabilities, the podium and the pre-qualifying picture.

## Retraining

```bash
python train_model.py                           # default: logistic regression, holdout on last complete season
python train_model.py --algorithm rf            # random forest
python train_model.py --algorithm hgb           # gradient boosting
python backtest.py                              # refresh models/backtest_results.json for the web app
```

Retrain after a few races of a new season, especially when regulations change. The bundle records the
scikit-learn version it was trained with and warns if a different version is installed.
