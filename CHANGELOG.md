# Changelog

## 1.1.0 (2026-09-05)

### Added
- Point-in-time race contexts (`history.py`): standings computed from results before each race, form rolling
  over from the previous season, entry lists from qualifying. Used identically by predictions, training and backtests.
- Walk-forward backtester (`backtest.py`) with pole-sitter and points-leader baselines; results feed the web app.
- Real probabilities: win probabilities sum to 100%, podium probabilities from a Plackett-Luce model.
- New factors: reliability, teammate comparison, sprint results, race-day weather from Open-Meteo.
- AI analyst (`--ai`): Claude reads the pre-race briefing and returns probabilities plus a written analysis.
- AI verdict (`--verdict`): average of the statistical, ML and AI probabilities.
- Streamlit web app (`app.py`) with prediction, model accuracy and about tabs.
- CLI: `--race`, `--season`, `--list-races`, `--all`, `--json`, `--compare`, `--no-weather`, `--version`.
- Test suite (80 tests) on a synthetic season, no network needed.

### Changed
- ML model retrained on point-in-time features with qualifying-masked copies; logistic regression replaces the
  random forest. Honest backtest accuracy replaces the previous 99.7% figure, which came from label leakage.
- Statistical weights re-tuned on the backtest (qualifying 45%); softmax temperature calibrated to 8.
- Model bundle saved with joblib including feature names, scikit-learn version and holdout metrics.
- Library modules no longer configure logging; the CLI does (`--debug` for details).

### Fixed
- Jolpica API responses are capped at 100 rows: season results, qualifying and sprints are now paged and merged
  by round. Previously "recent form" used the first five races of the season and circuit history returned
  1970s data that the recency filter then discarded.
- Lapped cars no longer count as retirements.
- Rate limiting and 429 handling for the Jolpica API.
