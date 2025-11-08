"""Command-line interface for F1 Race Predictor."""

import argparse
import sys
from typing import Optional

from f1_predictor.engine import PredictionEngine
from f1_predictor.models import PredictionError


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog='f1-predictor',
        description='Predict F1 race winners based on historical data and current form',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  f1-predictor                    # Predict next race (default)
  f1-predictor --next             # Predict next race (explicit)
  f1-predictor --top 5            # Show top 5 predictions
  f1-predictor --verbose          # Show detailed factor analysis
  f1-predictor --no-cache         # Disable caching
  f1-predictor --verbose --top 5  # Combine options
        """
    )
    
    # Race selection (mutually exclusive)
    race_group = parser.add_mutually_exclusive_group()
    race_group.add_argument(
        '--next',
        action='store_true',
        default=True,
        help='Predict next scheduled race (default behavior)'
    )
    race_group.add_argument(
        '--race',
        type=str,
        metavar='RACE_NAME',
        help='Predict specific upcoming race (not yet implemented)'
    )
    
    # Prediction options
    parser.add_argument(
        '--top',
        type=int,
        default=3,
        metavar='N',
        help='Number of top predictions to show (default: 3)'
    )
    
    # Output options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed factor analysis and scoring breakdown'
    )
    
    # Cache options
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching and fetch fresh data from API'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.top < 1:
        parser.error("--top must be at least 1")
    
    if args.top > 20:
        parser.error("--top cannot exceed 20 (grid size)")
    
    if args.race:
        parser.error("--race flag is not yet implemented. Use --next for next race prediction.")
    
    return args


def main() -> int:
    """
    Main entry point for CLI.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    engine = None
    
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Initialize prediction engine with configuration
        use_cache = not args.no_cache
        engine = PredictionEngine(
            use_cache=use_cache,
            top_n=args.top,
            verbose=args.verbose
        )
        
        # Show initial message
        print("F1 Race Predictor")
        print("=" * 65)
        print()
        
        # Generate prediction
        result = engine.predict_next_race()
        
        # Format and display results
        formatted_output = engine.format_result(result)
        print(formatted_output)
        
        return 0
        
    except PredictionError as e:
        # Handle prediction-specific errors
        if engine:
            print("\n" + engine.format_error(e), file=sys.stderr)
        else:
            print(f"\nERROR: {e.message}", file=sys.stderr)
        return 1
        
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print("\n\nPrediction cancelled by user.", file=sys.stderr)
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        # Handle unexpected errors
        print("\n" + "=" * 65, file=sys.stderr)
        print("UNEXPECTED ERROR", file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        print(f"\n{str(e)}\n", file=sys.stderr)
        print("Please report this issue if it persists.", file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
