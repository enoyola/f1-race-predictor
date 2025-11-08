"""Result formatter for F1 Race Predictor."""

from typing import List
from f1_predictor.models import PredictionResult, DriverPrediction


class ResultFormatter:
    """Formats prediction results for display."""
    
    def format_prediction(self, result: PredictionResult, verbose: bool = False) -> str:
        """
        Format predictions for console output.
        
        Args:
            result: The prediction result to format
            verbose: Whether to show detailed factor breakdowns
            
        Returns:
            Formatted string ready for display
        """
        output = []
        
        # Header
        output.append("F1 Race Winner Prediction")
        output.append("═" * 65)
        
        # Race information
        race = result.race
        output.append(f"Race: {race.race_name}")
        output.append(f"Circuit: {race.circuit.circuit_name}")
        output.append(f"Location: {race.circuit.location}, {race.circuit.country}")
        output.append(f"Date: {race.date.strftime('%B %d, %Y')}")
        output.append("")
        
        # Predictions
        output.append(f"TOP {len(result.predictions)} PREDICTIONS:")
        output.append("─" * 65)
        
        for i, prediction in enumerate(result.predictions, 1):
            output.append(self._format_single_prediction(i, prediction, verbose))
            if i < len(result.predictions):
                output.append("")
        
        # Footer
        output.append("─" * 65)
        output.append(f"Prediction generated: {result.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        output.append(f"Data sources: {', '.join(result.data_sources)}")
        output.append(f"Data completeness: {result.data_completeness * 100:.1f}%")
        
        return "\n".join(output)
    
    def _format_single_prediction(self, rank: int, prediction: DriverPrediction, verbose: bool) -> str:
        """Format a single driver prediction."""
        lines = []
        
        # Driver header
        driver_name = f"{prediction.driver.forename} {prediction.driver.surname}"
        team_name = prediction.constructor.name
        lines.append(f"{rank}. {driver_name} ({team_name})")
        lines.append(f"   Confidence: {prediction.confidence:.1f}%")
        lines.append("")
        
        # Key factors
        if verbose:
            lines.append("   Detailed Factors:")
            lines.append(self.format_factors(prediction))
        else:
            lines.append("   Key Factors:")
            for reason in prediction.reasoning:
                lines.append(f"   • {reason}")
        
        return "\n".join(lines)
    
    def format_factors(self, prediction: DriverPrediction) -> str:
        """
        Format individual factor scores for detailed display.
        
        Args:
            prediction: The driver prediction containing factor scores
            
        Returns:
            Formatted string showing all factor scores
        """
        lines = []
        
        # Define factor display names and order
        factor_names = {
            'championship': 'Championship Position',
            'form': 'Recent Form',
            'team': 'Team Performance',
            'qualifying': 'Qualifying Position',
            'circuit': 'Circuit History'
        }
        
        for key, display_name in factor_names.items():
            if key in prediction.factors:
                score = prediction.factors[key]
                lines.append(f"   • {display_name}: {score:.1f}/100")
        
        return "\n".join(lines)
    
    def format_table(self, predictions: List[DriverPrediction]) -> str:
        """
        Format predictions as ASCII table.
        
        Args:
            predictions: List of driver predictions to format
            
        Returns:
            Formatted ASCII table string
        """
        # Table header
        lines = []
        lines.append("┌──────┬─────────────────────────┬──────────────────────┬────────────┐")
        lines.append("│ Rank │ Driver                  │ Team                 │ Confidence │")
        lines.append("├──────┼─────────────────────────┼──────────────────────┼────────────┤")
        
        # Table rows
        for i, prediction in enumerate(predictions, 1):
            driver_name = f"{prediction.driver.forename} {prediction.driver.surname}"
            team_name = prediction.constructor.name
            confidence = f"{prediction.confidence:.1f}%"
            
            # Truncate if needed to fit column widths
            driver_name = driver_name[:23].ljust(23)
            team_name = team_name[:20].ljust(20)
            confidence = confidence.rjust(10)
            
            lines.append(f"│  {i:<2}  │ {driver_name} │ {team_name} │ {confidence} │")
        
        # Table footer
        lines.append("└──────┴─────────────────────────┴──────────────────────┴────────────┘")
        
        return "\n".join(lines)
