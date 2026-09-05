"""
Walk-forward backtest of the predictors against real results.

    python backtest.py                     # last three seasons
    python backtest.py --seasons 2022-2026
    python backtest.py --algorithm rf --no-weather

Writes models/backtest_results.json (used by the Streamlit app) and prints a
summary table with two naive baselines for comparison.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from f1_predictor.backtest import format_report, run_backtest
from f1_predictor.cache import DataCache
from f1_predictor.data_fetcher import F1DataFetcher
from f1_predictor.history import HistoricalDataStore
from f1_predictor.weather import WeatherClient

DEFAULT_OUTPUT = "models/backtest_results.json"


def parse_seasons(text: str):
    if "-" in text:
        a, b = text.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the F1 predictor")
    parser.add_argument("--seasons", default=None, help="Test seasons, e.g. 2023-2026 (default: last three seasons)")
    parser.add_argument("--min-train-season", type=int, default=2020, help="Earliest season used to train the ML model")
    parser.add_argument("--algorithm", choices=["logreg", "rf", "hgb"], default="logreg")
    parser.add_argument("--no-ml", action="store_true")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--ai", action="store_true", help="Also score the Claude AI analyst (one API call per race, costs money)")
    parser.add_argument("--temperature", type=float, default=None, help="Override the statistical softmax temperature")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--cache-dir", default=".f1_cache")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.debug else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cache = DataCache(args.cache_dir)
    fetcher = F1DataFetcher(cache=cache)
    weather = None if args.no_weather else WeatherClient(cache=cache)
    store = HistoricalDataStore(fetcher, weather)

    current = fetcher.get_current_season()
    seasons = parse_seasons(args.seasons) if args.seasons else list(range(current - 2, current + 1))

    started = datetime.now()
    progress = None if args.quiet else (lambda msg: print(f"  {msg}"))
    print(f"Backtesting seasons {seasons[0]}-{seasons[-1]}...")
    extra = {}
    if args.ai:
        from f1_predictor.ai_analyzer import AIPredictionAnalyzer
        ai = AIPredictionAnalyzer(cache=cache)
        if not ai.model_loaded:
            print(f"AI analyst unavailable: {ai.load_error}", file=sys.stderr)
            return 1
        extra["ai"] = ai
    report = run_backtest(
        store, seasons,
        extra_analyzers=extra,
        include_ml=not args.no_ml,
        algorithm=args.algorithm,
        use_weather=not args.no_weather,
        min_train_season=args.min_train_season,
        progress=progress,
        temperature=args.temperature,
    )
    print()
    print(format_report(report))
    print(f"\nAPI requests: {fetcher.request_count}, elapsed {(datetime.now() - started).seconds}s")

    if not args.no_save:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=1)
        print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
