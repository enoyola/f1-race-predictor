"""
Train the ML model for F1 race prediction.

Builds point-in-time features for every completed race in the chosen seasons,
reports out-of-sample accuracy on a holdout season, then retrains on all
seasons and saves the model bundle.

    python train_model.py                       # seasons 2020-current, holdout = last complete season
    python train_model.py --seasons 2021-2026 --holdout 2025
    python train_model.py --algorithm rf        # random forest instead of logistic regression
"""

import argparse
import logging
import sys
from datetime import datetime

from f1_predictor.backtest import score_predictions
from f1_predictor.cache import DataCache
from f1_predictor.data_fetcher import F1DataFetcher
from f1_predictor.features import FEATURE_NAMES
from f1_predictor.history import HistoricalDataStore
from f1_predictor.training import (
    DEFAULT_MODEL_PATH, build_training_rows, make_bundle, rows_to_arrays, save_bundle, train_models,
)
from f1_predictor.weather import WeatherClient

logger = logging.getLogger("train_model")


def parse_seasons(text: str):
    if "-" in text:
        a, b = text.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def evaluate_holdout(store, rows, holdout: int, algorithm: str, use_weather: bool) -> dict:
    """Train on seasons before the holdout, score every holdout race."""
    from f1_predictor.backtest import ModelMetrics
    from f1_predictor.ml_analyzer import MLPredictionAnalyzer

    train_rows = [r for r in rows if r.season < holdout]
    if len(train_rows) < 200:
        logger.warning("Not enough rows before %s to evaluate a holdout", holdout)
        return {}
    X, y_win, y_podium = rows_to_arrays(train_rows)
    win_model, podium_model = train_models(X, y_win, y_podium, algorithm=algorithm)
    analyzer = MLPredictionAnalyzer(bundle=make_bundle(win_model, podium_model, [], len(train_rows), algorithm=algorithm))

    totals = ModelMetrics()
    data = store.load_season(holdout, with_qualifying=True)
    for rnd in data.completed_rounds:
        context = store.build_context(holdout, rnd, include_actual=True, use_weather=use_weather)
        preds = analyzer.analyze(context)
        totals.add(score_predictions(preds, context.actual_results or []))
    metrics = totals.to_dict()
    metrics["holdout_season"] = holdout
    metrics["train_rows"] = len(train_rows)
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the F1 predictor ML model")
    parser.add_argument("--seasons", default=None, help="Seasons to train on, e.g. 2020-2026 or 2021,2022 (default: 2020-current)")
    parser.add_argument("--holdout", type=int, default=None, help="Season to evaluate on before the final fit (default: last complete season)")
    parser.add_argument("--no-holdout", action="store_true", help="Skip the holdout evaluation")
    parser.add_argument("--algorithm", choices=["logreg", "rf", "hgb"], default="logreg")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--no-augment", action="store_true", help="Do not add qualifying-masked copies of each race")
    parser.add_argument("--output", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cache-dir", default=".f1_cache")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.debug else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cache = DataCache(args.cache_dir)
    fetcher = F1DataFetcher(cache=cache)
    weather = None if args.no_weather else WeatherClient(cache=cache)
    store = HistoricalDataStore(fetcher, weather)

    current = fetcher.get_current_season()
    seasons = parse_seasons(args.seasons) if args.seasons else list(range(2020, current + 1))
    holdout = args.holdout or max(s for s in seasons if s < current) if any(s < current for s in seasons) else None
    if args.no_holdout:
        holdout = None

    print("F1 Race Predictor - model training")
    print("=" * 60)
    print(f"Seasons:   {seasons[0]}-{seasons[-1]}   holdout: {holdout or 'none'}   algorithm: {args.algorithm}")
    print(f"Features:  {len(FEATURE_NAMES)}  ({', '.join(FEATURE_NAMES)})")
    print()

    started = datetime.now()

    def progress(msg: str) -> None:
        print(f"  {msg}")

    print("Building point-in-time training rows...")
    rows = build_training_rows(store, seasons, use_weather=not args.no_weather, augment_masked=not args.no_augment, progress=progress)
    if len(rows) < 200:
        print(f"ERROR: only {len(rows)} training rows, need at least 200", file=sys.stderr)
        return 1
    wins = sum(r.won for r in rows)
    print(f"\n{len(rows)} rows from {len({(r.season, r.round) for r in rows})} races; {wins} winner rows; API requests: {fetcher.request_count}")

    metrics = {}
    if holdout:
        print(f"\nHoldout evaluation on {holdout} (trained on earlier seasons only):")
        metrics = evaluate_holdout(store, rows, holdout, args.algorithm, not args.no_weather)
        if metrics:
            print(f"  races {metrics['races']}  top-1 {metrics['top1_accuracy']*100:.1f}%  winner in top-3 {metrics['winner_in_top3']*100:.1f}%  "
                  f"podium precision {metrics['podium_precision']*100:.1f}%  log loss {metrics['log_loss']:.3f}")

    print("\nFitting final model on all seasons...")
    X, y_win, y_podium = rows_to_arrays(rows)
    win_model, podium_model = train_models(X, y_win, y_podium, algorithm=args.algorithm)
    bundle = make_bundle(win_model, podium_model, seasons, len(rows), metrics=metrics, algorithm=args.algorithm)
    path = save_bundle(bundle, args.output)

    if bundle.get("feature_importances"):
        print("\nFeature importance (win model):")
        for name, value in sorted(bundle["feature_importances"].items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {name:<20} {value*100:5.1f}%")

    print(f"\nSaved {path}  ({(datetime.now() - started).seconds}s)")
    print("Use it with:  python -m f1_predictor.cli --ml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
