"""Result formatting for the CLI: plain text and JSON."""

import textwrap
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from f1_predictor.features import FACTOR_LABELS
from f1_predictor.models import DriverPrediction, PredictionResult, RaceResult

RULE = "═" * 72
THIN = "─" * 72


class ResultFormatter:
    """Formats prediction results for display."""

    def format_prediction(self, result: PredictionResult, verbose: bool = False) -> str:
        race = result.race
        out: List[str] = []
        out.append(f"F1 Race Winner Prediction  ({result.model_name} model)")
        out.append(RULE)
        rounds = f" (Round {race.round}" + (f" of {race.total_rounds})" if race.total_rounds else ")")
        out.append(f"Race:      {race.race_name}{rounds}")
        out.append(f"Circuit:   {race.circuit.circuit_name}, {race.circuit.location}, {race.circuit.country}")
        out.append(f"Date:      {race.date.strftime('%A %d %B %Y')}")
        if race.is_sprint:
            out.append("Format:    Sprint weekend")
        quali = "available" if result.qualifying_available else "not yet available"
        pole = next((p for p in result.predictions if p.grid_position == 1), None)
        if pole is not None:
            quali += f" (pole: {pole.driver.surname})"
        out.append(f"Qualifying: {quali}")
        if result.weather is not None:
            out.append(f"Weather:   {result.weather.describe()}")
        for note in result.notes:
            out.append(f"Note:      {note}")
        out.append("")

        if result.analysis:
            out.append("ANALYSIS")
            out.append(THIN)
            out.append(textwrap.fill(result.analysis, width=72, replace_whitespace=False))
            out.append("")

        out.append(f"TOP {len(result.predictions)} PREDICTIONS")
        out.append(THIN)
        for i, prediction in enumerate(result.predictions, 1):
            out.append(self._format_single_prediction(i, prediction, verbose))
            if i < len(result.predictions):
                out.append("")
        out.append(THIN)

        if result.actual_results:
            out.append(self.format_actual(result.actual_results))
            out.append(THIN)

        out.append(f"Generated: {result.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        out.append(f"Data:      {', '.join(result.data_sources)}  |  completeness {result.data_completeness * 100:.0f}%")
        return "\n".join(out)

    def _format_single_prediction(self, rank: int, prediction: DriverPrediction, verbose: bool) -> str:
        lines: List[str] = []
        grid = f"  grid P{prediction.grid_position}" if prediction.grid_position else ""
        lines.append(
            f"{rank}. {prediction.driver.full_name} ({prediction.constructor.name}){grid}"
        )
        lines.append(
            f"   Win {prediction.win_probability * 100:5.1f}%   Podium {prediction.podium_probability * 100:5.1f}%"
            f"   Score {prediction.score:.1f}"
        )
        if verbose:
            lines.append("   Factors:")
            lines.append(self.format_factors(prediction))
            if prediction.reasoning:
                lines.append("   Reasoning:")
                for reason in prediction.reasoning:
                    lines.append(f"   • {reason}")
        else:
            for reason in prediction.reasoning[:4]:
                lines.append(f"   • {reason}")
        return "\n".join(lines)

    def format_factors(self, prediction: DriverPrediction) -> str:
        lines = []
        for key, label in FACTOR_LABELS.items():
            if key in prediction.factors:
                lines.append(f"   • {label:<22} {prediction.factors[key]:5.1f}/100")
        return "\n".join(lines)

    def format_actual(self, results: List[RaceResult]) -> str:
        top = sorted(results, key=lambda r: r.position)[:3]
        podium = ", ".join(f"P{r.position} {r.driver.surname}" for r in top)
        return f"Actual result: {podium}"

    def format_comparison(self, results: Dict[str, PredictionResult], top_n: int = 5) -> str:
        """Side-by-side table of several models' top predictions."""
        names = list(results.keys())
        first = results[names[0]]
        out: List[str] = []
        out.append(f"Model comparison: {first.race.race_name} {first.race.season}")
        out.append(RULE)
        header = "Rank".ljust(6) + "".join(name.ljust(32) for name in names)
        out.append(header)
        out.append(THIN)
        for i in range(top_n):
            row = f"{i + 1}.".ljust(6)
            for name in names:
                preds = results[name].predictions
                if i < len(preds):
                    p = preds[i]
                    row += f"{p.driver.surname:<12} {p.win_probability * 100:5.1f}% / {p.podium_probability * 100:5.1f}%".ljust(32)
                else:
                    row += "".ljust(32)
            out.append(row)
        out.append(THIN)
        out.append("Columns show win % / podium %")
        if first.actual_results:
            out.append(self.format_actual(first.actual_results))
        analysis = next((r.analysis for r in results.values() if r.analysis), None)
        if analysis:
            out.append("")
            out.append("AI analysis")
            out.append(THIN)
            out.append(textwrap.fill(analysis, width=72 if len(names) < 3 else 96, replace_whitespace=False))
        return "\n".join(out)

    def format_table(self, predictions: List[DriverPrediction]) -> str:
        lines = []
        lines.append("┌──────┬─────────────────────────┬──────────────────────┬────────┬────────┐")
        lines.append("│ Rank │ Driver                  │ Team                 │  Win   │ Podium │")
        lines.append("├──────┼─────────────────────────┼──────────────────────┼────────┼────────┤")
        for i, p in enumerate(predictions, 1):
            lines.append(
                f"│  {i:<2}  │ {p.driver.full_name[:23]:<23} │ {p.constructor.name[:20]:<20} │"
                f" {p.win_probability * 100:5.1f}% │ {p.podium_probability * 100:5.1f}% │"
            )
        lines.append("└──────┴─────────────────────────┴──────────────────────┴────────┴────────┘")
        return "\n".join(lines)


def result_to_dict(result: PredictionResult) -> Dict[str, Any]:
    """JSON-serializable representation of a prediction result."""
    race = result.race
    payload: Dict[str, Any] = {
        "model": result.model_name,
        "generated_at": result.generated_at.isoformat(),
        "race": {
            "season": race.season,
            "round": race.round,
            "total_rounds": race.total_rounds,
            "name": race.race_name,
            "date": race.date.isoformat(),
            "is_sprint": race.is_sprint,
            "circuit": asdict(race.circuit),
        },
        "qualifying_available": result.qualifying_available,
        "weather": asdict(result.weather) if result.weather else None,
        "data_sources": result.data_sources,
        "data_completeness": result.data_completeness,
        "notes": result.notes,
        "analysis": result.analysis,
        "components": result.components,
        "predictions": [
            {
                "rank": i,
                "driver_id": p.driver.driver_id,
                "code": p.driver.code,
                "driver": p.driver.full_name,
                "constructor": p.constructor.name,
                "constructor_id": p.constructor.constructor_id,
                "win_probability": round(p.win_probability, 4),
                "podium_probability": round(p.podium_probability, 4),
                "score": round(p.score, 2),
                "grid_position": p.grid_position,
                "factors": {k: round(v, 1) for k, v in p.factors.items()},
                "reasoning": p.reasoning,
            }
            for i, p in enumerate(result.predictions, 1)
        ],
    }
    if result.actual_results:
        payload["actual_results"] = [
            {"position": r.position, "driver_id": r.driver.driver_id, "driver": r.driver.full_name, "status": r.status}
            for r in sorted(result.actual_results, key=lambda r: r.position)
        ]
    return payload
