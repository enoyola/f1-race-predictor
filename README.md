# 🏎️ F1 Race Predictor

Predicts Formula 1 race winners and podiums from championship standings, recent form, qualifying,
sprint results, circuit history, reliability, teammate comparison and race-day weather.
Comes with a command-line tool, a Streamlit web app, a trainable machine-learning model and a
walk-forward backtester that reports how good the predictions actually are.

```
1. George Russell (Mercedes)  grid P2
   Win  39.5%   Podium  82.6%
   • Championship: P2, 183 pts, 2 win(s) [84/100]
   • Qualifying: P2 on the grid [90/100]
   ...
```

## ✨ Features

- **Real probabilities**: win probabilities sum to 100% across the field, podium probabilities sum to 300%
- **Three models**: a transparent weighted-factor model, a logistic-regression model trained on 2020 onwards, and an
  AI analyst (Claude) that reads the same briefing and writes an analysis, plus an **AI verdict** that blends them
- **Point-in-time data**: every prediction, training row and backtest only uses information available before the race
- **Backtesting**: `backtest.py` replays past seasons and compares both models with two naive baselines
- **Web app**: pick any race of any recent season, compare models, browse the field, see why each driver ranks where they do
- **Any race**: next race by default, or any round by number or name, past or future
- **Weather aware**: Open-Meteo forecast for upcoming races, archive for past ones; wet races weight the grid less
- **Sprint weekends**: sprint points feed the standings and the sprint result stands in for qualifying when needed
- **Caching**: API responses cached locally; completed seasons are kept for 30 days

## 📦 Installation

Python 3.9 or newer.

```bash
git clone https://github.com/enoyola/f1-race-predictor.git
cd f1-race-predictor
python -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -r requirements.txt                      # CLI + web app
```

Or install as a package (CLI only, add `[ui]` for the web app):

```bash
pip install -e ".[ui,ai]"
f1-predictor
```

## 🎮 Command line

```bash
python -m f1_predictor.cli                       # next race, statistical model, top 3
python -m f1_predictor.cli --ml                  # machine-learning model
python -m f1_predictor.cli --ai                  # Claude's analyst prediction with written analysis
python -m f1_predictor.cli --verdict             # every model plus a blended verdict
python -m f1_predictor.cli --compare --top 5     # all available models side by side
python -m f1_predictor.cli --race monza          # a race this season by name fragment
python -m f1_predictor.cli --race 5 --season 2025 --verbose   # a past race, with the real result
python -m f1_predictor.cli --list-races          # calendar with completed/upcoming status
python -m f1_predictor.cli --all --json          # the whole field as JSON
```

| Option | Description |
|--------|-------------|
| `--next` | Predict the next scheduled race (default) |
| `--race ROUND_OR_NAME` | Predict a specific race by round number or name fragment |
| `--season YEAR` | Season for `--race` and `--list-races` (default: current) |
| `--list-races` | Print the season calendar and exit |
| `--top N` / `--all` | Number of drivers to show (default 3) or the whole field |
| `--ml` | Use the trained machine-learning model |
| `--ai` | Ask Claude for an analyst prediction (needs `ANTHROPIC_API_KEY`) |
| `--verdict` | Blend statistical, ML and AI into one verdict |
| `--compare` | All available models side by side |
| `--verbose` | Factor scores and full reasoning per driver |
| `--json` | JSON output |
| `--no-cache` / `--no-weather` | Bypass the cache / skip the weather lookup |

## 🖥️ Web app

```bash
streamlit run app.py
```

Three tabs:

- **Prediction**: season and race picker (defaults to the next race), model selector (statistical, machine
  learning, AI analyst, AI verdict, or compare all), win and podium probability chart, the full field with
  factor scores, the AI analysis when applicable, and an explanation per driver. Completed races show the
  real result next to the prediction that would have been made before the race.
- **Model accuracy**: the backtest results, per season and per race, against the baselines.
- **About**: methodology, the trained model's metadata and feature importances.

To deploy on [Streamlit Community Cloud](https://streamlit.io/cloud), point it at this repository with
`app.py` as the entry point. The trained model and backtest results are committed, so no training step is
needed on the server. Add `ANTHROPIC_API_KEY` to the app's secrets to enable the AI analyst there.

## 🧮 How predictions work

For any race the predictor builds a **pre-race context** from data available before the start:

| Factor | Source | Statistical weight |
|--------|--------|-------------------|
| Qualifying position | Grid from qualifying (sprint result if the grid is not set yet) | 45% |
| Championship position | Standings computed from results and sprints so far (last season's at round 1) | 15% |
| Recent form | Last 5 races, rolling over from the previous season | 12% |
| Team performance | Constructor standings | 10% |
| Teammate comparison | Average finishing gap to the teammate over 10 races | 7% |
| Circuit history | Results at this circuit in the last 5 seasons | 6% |
| Reliability | Retirements in the last 10 races | 5% |

**Statistical model**: the weighted average of those 0-100 scores becomes a win probability through a
softmax (temperature calibrated by the backtest) and a podium probability through a Plackett-Luce model.
In the wet the qualifying weight drops from 45% to 30%.

**Machine-learning model**: two logistic-regression classifiers (win, podium) on 17 features, including
the ones above plus raw grid position, points shares, a wet-race flag and a sprint-weekend flag. Every
race is also included with qualifying hidden so the model works before Saturday. Probabilities are
normalized across the field. Random forest and gradient boosting are available with `--algorithm`.

**AI analyst**: Claude receives the same briefing as JSON (grid, standings, form, circuit history, reliability,
teammate gap, weather, plus the two models' picks) and returns a win and podium probability for every driver
with a three-paragraph analysis and a note per driver. It uses the Anthropic API with structured JSON output;
one call per race, cached for six hours. Set `ANTHROPIC_API_KEY` (or log in with the `ant` CLI) to enable it,
and `F1_AI_MODEL` to pick a model other than the default `claude-opus-5`. Without credentials the CLI and
app fall back to the statistical model and say so.

**AI verdict**: the average of the available models' win and podium probabilities, renormalized, with the AI
analysis attached and each model's vote shown per driver.

**Which one should I pick?**

| Option | What it is | Use it when |
|--------|------------|-------------|
| Statistical | Weighted factor scores, no training | You want the best winner accuracy and full transparency |
| Machine learning | Logistic regression trained on 2020 onwards | You want a data-driven second opinion, best at podiums |
| AI analyst | Claude's own probabilities and written analysis | You want the reasoning and circuit-specific judgement |
| AI verdict | Plain average of the three above | You want one number to act on |
| Compare all | Every model side by side | You want to see where they disagree |

All models read the same context, so the backtest numbers below are what to expect before a real race.
The AI analyst can be scored too with `python backtest.py --ai`, at one API call per race.

## 📊 How accurate is it?

Walk-forward backtest over the 82 races of 2023 to mid-2026. Each race is predicted with pre-race data
only; the ML model is retrained for each season on earlier seasons only. Reproduce with `python backtest.py`.

| Model | Winner picked | Winner in top 3 | Podium precision | Log loss |
|-------|--------------:|----------------:|-----------------:|---------:|
| Statistical | **61.0%** | **91.5%** | 66.3% | **1.086** |
| Machine learning (logistic regression) | 51.2% | 86.6% | **67.5%** | 1.240 |
| Baseline: pole sitter wins | 63.4% | 90.2% | 65.9% | – |
| Baseline: points leader wins | 45.1% | 80.5% | 58.1% | – |

Winner accuracy by season for the statistical model: 2023 82%, 2024 46%, 2025 58%, 2026 58% (12 races).
2023 was a one-driver season; 2024 to 2026 are what a competitive era looks like.

What this says: the grid is by far the strongest signal, and simply picking the pole sitter is hard to
beat on winner accuracy. The models earn their keep on the probabilities: a log loss of 1.09 means the
real winner was given about 34% on average, and the winner is in the predicted top three in over 90% of
races. Podium precision of two thirds means two of the three predicted podium drivers usually finish there.

## 🤖 Training the model

```bash
python train_model.py                       # 2020 to current season, holdout on the last complete season
python train_model.py --seasons 2021-2026 --holdout 2025 --algorithm rf
python backtest.py --seasons 2022-2026      # evaluate and refresh models/backtest_results.json
```

Training fetches about 100 API pages on the first run (then cached) and takes a few seconds to fit.
The bundle in `models/f1_predictor.joblib` stores the two models, the feature list, the scikit-learn
version, the holdout metrics and feature importances. If the feature list changes, the ML analyzer
falls back to the statistical model and asks you to retrain.

See [ML_COMPARISON.md](ML_COMPARISON.md) for the two models side by side.

## 🌐 Data sources

- [Jolpica F1 API](https://api.jolpi.ca/ergast/) for schedules, results, qualifying and sprints. Responses
  are capped at 100 rows, so season endpoints are paged and merged. Unauthenticated use is limited to
  4 requests/second and 500 requests/hour; the fetcher rate-limits itself and honours 429 responses.
- [Open-Meteo](https://open-meteo.com/) for race-day precipitation and temperature (no key needed).
- [Anthropic API](https://docs.anthropic.com/) for the AI analyst (optional, needs a key).

Cache directory: `.f1_cache/` (delete it or use `--no-cache` to refetch).

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite runs against a small synthetic season, so it needs no network access.

## 📁 Project structure

```
f1_predictor/
├── cli.py            Command-line interface
├── engine.py         Orchestrates data, context, models and formatting
├── data_fetcher.py   Jolpica API client with pagination, retries and caching
├── weather.py        Open-Meteo client
├── history.py        Season loading, standings computation, point-in-time contexts
├── features.py       Scoring functions and the feature vector
├── analyzer.py       Statistical model
├── ml_analyzer.py    Machine-learning model
├── ai_analyzer.py    AI analyst (Claude) with cached structured responses
├── training.py       Training-set construction, model fitting, bundle I/O
├── backtest.py       Walk-forward evaluation and baselines
├── probability.py    Softmax, Plackett-Luce, normalization helpers
├── formatter.py      Text and JSON output
├── cache.py          JSON file cache with TTLs
└── models.py         Dataclasses
app.py                Streamlit web app
train_model.py        Train and save the ML model
backtest.py           Run the backtest
models/               Trained model bundle and backtest results
tests/                pytest suite
```

## 📄 License

MIT
