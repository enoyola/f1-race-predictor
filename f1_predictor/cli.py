"""Command-line interface for F1 Race Predictor."""

import argparse
import json
import logging
import sys

from f1_predictor import __version__
from f1_predictor.engine import PredictionEngine
from f1_predictor.models import PredictionError


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='f1-predictor',
        description='Predict F1 race winners from standings, form, qualifying, circuit history and weather',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  f1-predictor                          # Predict the next race (statistical model)
  f1-predictor --ml                     # Use the trained machine-learning model
  f1-predictor --ai                     # Claude's analyst prediction with written analysis
  f1-predictor --verdict                # Blend of statistical, ML and AI
  f1-predictor --compare --top 5        # Every available model side by side
  f1-predictor --race monza             # A specific race this season (name fragment)
  f1-predictor --race 5 --season 2025   # Round 5 of 2025, shows the actual result too
  f1-predictor --list-races             # Calendar with completed/upcoming status
  f1-predictor --all --json             # Whole field as JSON
        """,
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    race_group = parser.add_mutually_exclusive_group()
    race_group.add_argument('--next', action='store_true', help='Predict the next scheduled race (default)')
    race_group.add_argument('--race', type=str, metavar='ROUND_OR_NAME',
                            help='Predict a specific race: round number or name fragment (e.g. "monza", "British")')
    race_group.add_argument('--list-races', action='store_true', help='List the season calendar and exit')
    parser.add_argument('--season', type=int, metavar='YEAR', help='Season for --race / --list-races (default: current)')

    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument('--top', type=int, default=3, metavar='N', help='Number of predictions to show (default: 3)')
    count_group.add_argument('--all', action='store_true', help='Show the whole field')

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--ml', action='store_true', help='Use the machine-learning model')
    model_group.add_argument('--ai', action='store_true', help='Ask Claude for an analyst prediction (needs ANTHROPIC_API_KEY)')
    model_group.add_argument('--verdict', action='store_true', help='Blend statistical, ML and AI predictions into one verdict')
    model_group.add_argument('--compare', action='store_true', help='Show every available model side by side')
    parser.add_argument('--model-path', type=str, metavar='PATH', help='Path to a trained model bundle')

    parser.add_argument('--verbose', action='store_true', help='Show factor scores and full reasoning')
    parser.add_argument('--json', action='store_true', help='Output JSON instead of text')
    parser.add_argument('--no-cache', action='store_true', help='Bypass the local cache')
    parser.add_argument('--no-weather', action='store_true', help='Skip the weather lookup')
    parser.add_argument('--debug', action='store_true', help='Show debug logging')

    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.top > 30:
        parser.error("--top cannot exceed 30")
    return args


def configure_logging(debug: bool) -> None:
    level = logging.INFO if debug else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s %(name)s: %(message)s', stream=sys.stderr)


def list_races(engine: PredictionEngine, season: int) -> str:
    schedule = engine.get_schedule(season)
    completed = set(engine.completed_rounds(season))
    lines = [f"{season} calendar ({len(schedule)} rounds)", "─" * 60]
    for race in schedule:
        status = "done" if race.round in completed else "upcoming"
        sprint = " (sprint)" if race.is_sprint else ""
        lines.append(f"{race.round:>2}. {race.date.strftime('%Y-%m-%d')}  {race.race_name:<32}{sprint} [{status}]")
    return "\n".join(lines)


def main(argv=None) -> int:
    engine = None
    try:
        args = parse_arguments(argv)
        configure_logging(args.debug)

        engine = PredictionEngine(
            use_cache=not args.no_cache,
            top_n=None if args.all else args.top,
            verbose=args.verbose,
            use_ml=args.ml,
            model_path=args.model_path,
            use_weather=not args.no_weather,
        )

        season = args.season or engine.current_season()

        if args.list_races:
            print(list_races(engine, season))
            return 0

        if args.race:
            race = engine.resolve_race(args.race, season)
        else:
            race = engine.get_next_race()

        if not args.json:
            print("F1 Race Predictor")
            print("=" * 72)
            print()

        if args.compare or args.verdict:
            results = engine.verdict(race.season, race.round) if args.verdict else engine.compare(race.season, race.round)
            if args.json:
                print(json.dumps({name: engine.to_dict(r) for name, r in results.items()}, indent=2))
            elif args.verdict:
                print(engine.format_comparison({k: v for k, v in results.items() if k != "verdict"}))
                print()
                print(engine.format_result(results["verdict"]))
            else:
                print(engine.format_comparison(results))
            return 0

        result = engine.predict_race(race.season, race.round, "ai" if args.ai else None)
        if args.json:
            print(json.dumps(engine.to_dict(result), indent=2))
        else:
            print(engine.format_result(result))
        return 0

    except PredictionError as e:
        if engine:
            print("\n" + engine.format_error(e), file=sys.stderr)
        else:
            print(f"\nERROR: {e.message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nPrediction cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:  # pragma: no cover - last resort
        print("\n" + "=" * 65, file=sys.stderr)
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        print("Run with --debug for details.", file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        logging.getLogger(__name__).debug("Unexpected error", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
