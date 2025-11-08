# F1 Race Predictor

A Python command-line tool that predicts Formula 1 race winners using historical performance data, qualifying results, and statistical analysis.

## Features

- **Data-Driven Predictions**: Analyzes multiple factors including driver form, team performance, qualifying positions, and circuit history
- **Confidence Scores**: Provides confidence percentages for each prediction based on data quality and factor weights
- **Smart Caching**: Caches API responses locally to improve performance and reduce API calls
- **Detailed Analysis**: Optional verbose mode shows factor breakdowns and scoring details
- **Easy to Use**: Simple command-line interface with sensible defaults

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Install from Source

1. Clone or download this repository:
```bash
git clone <repository-url>
cd f1-race-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the predictor:
```bash
python -m f1_predictor.cli
```

### Install as Package (Optional)

```bash
pip install -e .
```

After installation, you can run:
```bash
f1-predictor
```

## Usage

### Basic Usage

Predict the next scheduled F1 race:
```bash
python -m f1_predictor.cli
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--next` | Predict next scheduled race | Enabled by default |
| `--top N` | Show top N predictions | 3 |
| `--verbose` | Show detailed factor analysis | Disabled |
| `--no-cache` | Disable caching, fetch fresh data | Caching enabled |

### Examples

Show top 5 predictions:
```bash
python -m f1_predictor.cli --top 5
```

Show detailed factor breakdown:
```bash
python -m f1_predictor.cli --verbose
```

Get fresh data without using cache:
```bash
python -m f1_predictor.cli --no-cache
```

Combine multiple options:
```bash
python -m f1_predictor.cli --verbose --top 5 --no-cache
```

## Prediction Methodology

The predictor uses a weighted scoring model that combines multiple factors to generate predictions:

### Scoring Factors

1. **Championship Position (25%)**: Current driver standings in the championship
2. **Recent Form (20%)**: Performance in the last 5 races
3. **Team Performance (20%)**: Constructor championship standings
4. **Qualifying Position (20%)**: Grid position from qualifying session
5. **Circuit History (15%)**: Historical performance at the specific circuit

### Confidence Calculation

```
Final Score = (0.25 × Championship) + (0.20 × Form) + (0.20 × Team) + 
              (0.20 × Qualifying) + (0.15 × Circuit)

Confidence = Final Score × Data Completeness Factor
```

The confidence score is adjusted based on data availability. For example:
- If qualifying hasn't happened yet, confidence is reduced
- If circuit history is limited, confidence is adjusted accordingly
- New drivers or teams receive lower confidence scores

### Data Sources

- **Primary API**: [Jolpica F1 API](https://api.jolpi.ca/ergast) - A modern replacement for the discontinued Ergast API
- **Data Coverage**: Historical F1 data from 1950 to present
- **Update Frequency**: Real-time during race weekends

### Caching Strategy

To improve performance and reduce API load, the predictor caches data with the following TTL (Time To Live) values:

| Data Type | Cache Duration |
|-----------|----------------|
| Season results | 24 hours |
| Driver/Constructor standings | 6 hours |
| Qualifying results | 1 hour |
| Circuit history | 7 days |

Cache files are stored in the `.f1_cache/` directory.

## Output Format

### Standard Output

```
F1 Race Predictor
═════════════════════════════════════════════════════════════
Race: Monaco Grand Prix 2025
Circuit: Circuit de Monaco
Date: May 25, 2025

TOP 3 PREDICTIONS:
─────────────────────────────────────────────────────────────
1. Max Verstappen (Red Bull Racing)
   Confidence: 78.5%

2. Charles Leclerc (Ferrari)
   Confidence: 72.3%

3. Sergio Perez (Red Bull Racing)
   Confidence: 65.8%

─────────────────────────────────────────────────────────────
Prediction generated: 2025-05-24 14:30:00 UTC
Data sources: Jolpica F1 API
```

### Verbose Output

With `--verbose` flag, additional factor breakdowns are shown:

```
1. Max Verstappen (Red Bull Racing)
   Confidence: 78.5%
   
   Factor Breakdown:
   • Championship Position: #1 → 25.0 pts (weight: 25%)
   • Recent Form: 4 wins in last 5 races → 20.0 pts (weight: 20%)
   • Team Performance: Leading constructor → 18.5 pts (weight: 20%)
   • Qualifying: P1 → 20.0 pts (weight: 20%)
   • Circuit History: 2 wins at Monaco → 15.0 pts (weight: 15%)
```

## Troubleshooting

### Common Issues

#### "Network error: Unable to connect to API"

**Cause**: Cannot reach the Jolpica F1 API

**Solutions**:
- Check your internet connection
- Verify the API is accessible: `curl https://api.jolpi.ca/ergast/f1/current/next.json`
- Try again later if the API is temporarily down
- Use cached data if available (remove `--no-cache` flag)

#### "No qualifying data available yet"

**Cause**: Qualifying session hasn't occurred or data isn't published yet

**Solutions**:
- This is normal before qualifying happens
- The predictor will generate predictions using available data
- Confidence scores will be adjusted to reflect missing data
- Run again after qualifying completes for updated predictions

#### "Cache directory permission denied"

**Cause**: Cannot write to `.f1_cache/` directory

**Solutions**:
- Check directory permissions: `ls -la .f1_cache`
- Create directory manually: `mkdir .f1_cache`
- Run with `--no-cache` to disable caching

#### "Prediction timeout"

**Cause**: API requests taking too long

**Solutions**:
- Check your network speed
- Try using cached data (remove `--no-cache` flag)
- The predictor has a 10-second timeout per request with retry logic

### Data Quality Issues

If predictions seem inaccurate:

1. **Check data freshness**: Use `--no-cache` to fetch latest data
2. **Verify race weekend timing**: Predictions are most accurate after qualifying
3. **New season limitations**: Early season predictions have less historical data
4. **New drivers/teams**: Limited data results in lower confidence scores

### Getting Help

If you encounter persistent issues:

1. Check that you're using Python 3.8 or higher: `python --version`
2. Verify all dependencies are installed: `pip list`
3. Try running with `--verbose` for detailed output
4. Clear cache and try again: `rm -rf .f1_cache`

## Project Structure

```
f1-race-predictor/
├── f1_predictor/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── engine.py           # Prediction orchestrator
│   ├── data_fetcher.py     # API data retrieval
│   ├── cache.py            # Local caching system
│   ├── analyzer.py         # Prediction analysis
│   ├── formatter.py        # Output formatting
│   └── models.py           # Data models
├── .f1_cache/              # Cached API responses (auto-generated)
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation
└── README.md               # This file
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines.

## Attribution

- **F1 Data**: Provided by [Jolpica F1 API](https://api.jolpi.ca/ergast)
- **Original API**: Based on the Ergast Developer API structure
- **Data License**: Check Jolpica F1 API terms of use

## Limitations

- Predictions are statistical estimates and not guarantees
- Accuracy depends on data quality and availability
- Does not account for weather conditions, mechanical failures, or race incidents
- Limited effectiveness for new drivers or teams with minimal historical data
- Cannot predict unexpected events (crashes, safety cars, strategy variations)

## Future Enhancements

Potential improvements for future versions:

- Machine learning models for more sophisticated predictions
- Weather forecast integration
- Live updates during race weekends
- Web-based interface
- Historical accuracy tracking
- Driver-specific factors (tire strategy, wet weather performance)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
