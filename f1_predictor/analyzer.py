"""
Prediction analyzer module.

Analyzes F1 data and calculates prediction scores based on multiple factors
including driver form, team performance, circuit history, and qualifying results.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from f1_predictor.models import (
    Race, Driver, Constructor, RaceResult, QualifyingResult,
    DriverStanding, ConstructorStanding, DriverPrediction
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionAnalyzer:
    """
    Analyzes F1 data to generate race winner predictions.
    
    Uses a weighted scoring model combining multiple factors:
    - Driver Championship Position: 25%
    - Recent Race Results (last 5 races): 20%
    - Constructor Performance: 20%
    - Qualifying Position: 20%
    - Circuit-Specific History: 15%
    """
    
    # Scoring weights
    WEIGHT_CHAMPIONSHIP = 0.25
    WEIGHT_FORM = 0.20
    WEIGHT_TEAM = 0.20
    WEIGHT_QUALIFYING = 0.20
    WEIGHT_CIRCUIT = 0.15
    
    # Configuration
    RECENT_RACES_COUNT = 5  # Number of recent races to consider for form
    
    def __init__(self):
        """Initialize the prediction analyzer."""
        pass
    
    def calculate_driver_form(
        self,
        driver: Driver,
        recent_results: List[RaceResult]
    ) -> float:
        """
        Calculate driver's recent form score based on last N races.
        
        Scoring logic:
        - Win (1st): 25 points
        - Podium (2nd-3rd): 18-15 points
        - Points finish (4th-10th): 12-3 points
        - No points: 0 points
        - DNF/DNS: -2 points
        
        Args:
            driver: Driver to calculate form for
            recent_results: List of recent race results (should be last 5 races)
            
        Returns:
            Form score from 0-100
        """
        if not recent_results:
            return 50.0  # Neutral score if no data
        
        # Filter results for this driver and get most recent races
        driver_results = [
            r for r in recent_results
            if r.driver.driver_id == driver.driver_id
        ]
        
        # Sort by race date (most recent first) and take last N races
        driver_results.sort(key=lambda r: r.race.date, reverse=True)
        driver_results = driver_results[:self.RECENT_RACES_COUNT]
        
        if not driver_results:
            return 50.0  # Neutral score if no driver data
        
        # Calculate points based on positions
        total_points = 0.0
        for result in driver_results:
            if result.status != "Finished" and result.position > 10:
                # DNF or DNS
                total_points += -2
            elif result.position == 1:
                total_points += 25
            elif result.position == 2:
                total_points += 18
            elif result.position == 3:
                total_points += 15
            elif result.position == 4:
                total_points += 12
            elif result.position == 5:
                total_points += 10
            elif result.position == 6:
                total_points += 8
            elif result.position == 7:
                total_points += 6
            elif result.position == 8:
                total_points += 4
            elif result.position == 9:
                total_points += 2
            elif result.position == 10:
                total_points += 1
            else:
                total_points += 0
        
        # Normalize to 0-100 scale
        # Maximum possible: 25 * 5 = 125 points
        # Minimum possible: -2 * 5 = -10 points
        max_points = 25 * self.RECENT_RACES_COUNT
        min_points = -2 * self.RECENT_RACES_COUNT
        
        # Shift to 0-based scale and normalize
        normalized = ((total_points - min_points) / (max_points - min_points)) * 100
        
        return max(0.0, min(100.0, normalized))
    
    def calculate_team_performance(
        self,
        constructor: Constructor,
        standings: List[ConstructorStanding]
    ) -> float:
        """
        Calculate team performance score based on constructor standings.
        
        Scoring logic:
        - Position-based: 1st = 100, 2nd = 90, 3rd = 80, etc.
        - Adjusted by wins ratio
        
        Args:
            constructor: Constructor to calculate performance for
            standings: List of constructor standings
            
        Returns:
            Team performance score from 0-100
        """
        if not standings:
            return 50.0  # Neutral score if no data
        
        # Find constructor in standings
        constructor_standing = None
        for standing in standings:
            if standing.constructor.constructor_id == constructor.constructor_id:
                constructor_standing = standing
                break
        
        if not constructor_standing:
            return 50.0  # Neutral score if constructor not found
        
        # Base score from position (1st = 100, decreasing by 10 per position)
        num_teams = len(standings)
        position_score = max(0, 100 - ((constructor_standing.position - 1) * (100 / num_teams)))
        
        # Bonus for wins (up to 10% boost)
        total_wins = sum(s.wins for s in standings)
        if total_wins > 0:
            win_ratio = constructor_standing.wins / total_wins
            win_bonus = win_ratio * 10
        else:
            win_bonus = 0
        
        final_score = min(100.0, position_score + win_bonus)
        return final_score

    def calculate_circuit_advantage(
        self,
        driver: Driver,
        circuit: str,
        history: List[RaceResult]
    ) -> float:
        """
        Calculate driver's historical performance advantage at specific circuit.
        
        Scoring logic:
        - Wins at circuit: 30 points each
        - Podiums at circuit: 15 points each
        - Points finishes: 5 points each
        - Normalized to 0-100 scale
        
        Args:
            driver: Driver to calculate circuit advantage for
            circuit: Circuit ID
            history: List of historical race results at this circuit
            
        Returns:
            Circuit advantage score from 0-100
        """
        if not history:
            return 50.0  # Neutral score if no history
        
        # Filter results for this driver at this circuit
        driver_history = [
            r for r in history
            if r.driver.driver_id == driver.driver_id
        ]
        
        if not driver_history:
            return 50.0  # Neutral score if driver has no history at circuit
        
        # Calculate points based on historical performance
        total_points = 0.0
        wins = 0
        podiums = 0
        points_finishes = 0
        
        for result in driver_history:
            if result.position == 1:
                total_points += 30
                wins += 1
            elif result.position <= 3:
                total_points += 15
                podiums += 1
            elif result.position <= 10:
                total_points += 5
                points_finishes += 1
        
        # Normalize based on number of races at circuit
        # More races = more reliable data
        num_races = len(driver_history)
        if num_races > 0:
            avg_points = total_points / num_races
            # Scale to 0-100 (assuming max average of 30 points per race)
            normalized = (avg_points / 30) * 100
        else:
            normalized = 50.0
        
        return max(0.0, min(100.0, normalized))
    
    def calculate_qualifying_impact(self, qualifying_position: int) -> float:
        """
        Calculate impact of qualifying position on race outcome.
        
        Scoring logic:
        - Pole position (1st): 100 points
        - Front row (2nd): 90 points
        - Top 3: 80 points
        - Top 5: 70 points
        - Top 10: 50 points
        - Beyond top 10: decreasing score
        
        Args:
            qualifying_position: Grid position from qualifying (1-20)
            
        Returns:
            Qualifying impact score from 0-100
        """
        if qualifying_position <= 0:
            return 50.0  # Neutral score for invalid position
        
        if qualifying_position == 1:
            return 100.0
        elif qualifying_position == 2:
            return 90.0
        elif qualifying_position == 3:
            return 80.0
        elif qualifying_position <= 5:
            return 70.0
        elif qualifying_position <= 10:
            return 50.0
        else:
            # Decreasing score for positions beyond top 10
            score = max(0, 50 - ((qualifying_position - 10) * 5))
            return float(score)
    
    def calculate_championship_score(
        self,
        driver: Driver,
        standings: List[DriverStanding]
    ) -> float:
        """
        Calculate score based on driver's championship position.
        
        Scoring logic:
        - Position-based: 1st = 100, 2nd = 90, 3rd = 80, etc.
        - Adjusted by points gap to leader
        
        Args:
            driver: Driver to calculate championship score for
            standings: List of driver standings
            
        Returns:
            Championship score from 0-100
        """
        if not standings:
            return 50.0  # Neutral score if no data
        
        # Find driver in standings
        driver_standing = None
        for standing in standings:
            if standing.driver.driver_id == driver.driver_id:
                driver_standing = standing
                break
        
        if not driver_standing:
            return 50.0  # Neutral score if driver not found
        
        # Base score from position
        num_drivers = len(standings)
        position_score = max(0, 100 - ((driver_standing.position - 1) * (100 / num_drivers)))
        
        # Adjust by points gap to leader
        leader_points = standings[0].points if standings else 0
        if leader_points > 0:
            points_ratio = driver_standing.points / leader_points
            # Apply points ratio as a multiplier (0.5 to 1.0 range)
            points_multiplier = 0.5 + (points_ratio * 0.5)
            position_score *= points_multiplier
        
        return max(0.0, min(100.0, position_score))

    def combine_factors(
        self,
        factors: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Combine multiple scoring factors with weights into final score.
        
        Uses weighted average of all factors. Default weights:
        - Championship: 25%
        - Form: 20%
        - Team: 20%
        - Qualifying: 20%
        - Circuit: 15%
        
        Args:
            factors: Dictionary of factor names to scores (0-100)
            weights: Optional custom weights (defaults to class weights)
            
        Returns:
            Combined score from 0-100
        """
        if not factors:
            return 50.0
        
        # Use default weights if not provided
        if weights is None:
            weights = {
                'championship': self.WEIGHT_CHAMPIONSHIP,
                'form': self.WEIGHT_FORM,
                'team': self.WEIGHT_TEAM,
                'qualifying': self.WEIGHT_QUALIFYING,
                'circuit': self.WEIGHT_CIRCUIT
            }
        
        # Calculate weighted sum
        total_score = 0.0
        total_weight = 0.0
        
        for factor_name, score in factors.items():
            weight = weights.get(factor_name, 0.0)
            total_score += score * weight
            total_weight += weight
        
        # Normalize by total weight (in case not all factors present)
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 50.0
        
        return max(0.0, min(100.0, final_score))
    
    def calculate_confidence(
        self,
        combined_score: float,
        data_completeness: float
    ) -> float:
        """
        Calculate final confidence score adjusted for data completeness.
        
        Confidence is the combined score adjusted by how complete the data is.
        Missing data reduces confidence.
        
        Args:
            combined_score: Combined factor score (0-100)
            data_completeness: Data completeness factor (0-1)
            
        Returns:
            Confidence score from 0-100
        """
        # Adjust score by data completeness
        # Full data (1.0) = no adjustment
        # Partial data (0.5) = 50% confidence reduction
        confidence = combined_score * data_completeness
        
        return max(0.0, min(100.0, confidence))
    
    def generate_reasoning(
        self,
        driver: Driver,
        constructor: Constructor,
        factors: Dict[str, float],
        driver_standing: Optional[DriverStanding] = None,
        qualifying_position: Optional[int] = None,
        recent_results: Optional[List[RaceResult]] = None,
        circuit_history: Optional[List[RaceResult]] = None
    ) -> List[str]:
        """
        Generate human-readable reasoning for prediction.
        
        Creates a list of key factors that influenced the prediction,
        with specific statistics and context.
        
        Args:
            driver: Driver being predicted
            constructor: Driver's constructor
            factors: Dictionary of factor scores
            driver_standing: Driver's championship standing (optional)
            qualifying_position: Qualifying position (optional)
            recent_results: Recent race results (optional)
            circuit_history: Circuit-specific history (optional)
            
        Returns:
            List of reasoning strings
        """
        reasoning = []
        
        # Championship position
        if driver_standing and 'championship' in factors:
            reasoning.append(
                f"Championship Position: #{driver_standing.position} "
                f"({driver_standing.points} pts, {driver_standing.wins} wins) "
                f"[Score: {factors['championship']:.1f}]"
            )
        
        # Recent form
        if recent_results and 'form' in factors:
            driver_results = [
                r for r in recent_results
                if r.driver.driver_id == driver.driver_id
            ]
            driver_results.sort(key=lambda r: r.race.date, reverse=True)
            driver_results = driver_results[:self.RECENT_RACES_COUNT]
            
            if driver_results:
                wins = sum(1 for r in driver_results if r.position == 1)
                podiums = sum(1 for r in driver_results if r.position <= 3)
                
                if wins > 0:
                    reasoning.append(
                        f"Recent Form: {wins} win(s) in last {len(driver_results)} races "
                        f"[Score: {factors['form']:.1f}]"
                    )
                elif podiums > 0:
                    reasoning.append(
                        f"Recent Form: {podiums} podium(s) in last {len(driver_results)} races "
                        f"[Score: {factors['form']:.1f}]"
                    )
                else:
                    avg_position = sum(r.position for r in driver_results) / len(driver_results)
                    reasoning.append(
                        f"Recent Form: Avg position {avg_position:.1f} in last {len(driver_results)} races "
                        f"[Score: {factors['form']:.1f}]"
                    )
        
        # Team performance
        if 'team' in factors:
            reasoning.append(
                f"Team Performance: {constructor.name} "
                f"[Score: {factors['team']:.1f}]"
            )
        
        # Qualifying position
        if qualifying_position and 'qualifying' in factors:
            position_text = "Pole position" if qualifying_position == 1 else f"P{qualifying_position}"
            reasoning.append(
                f"Qualifying: {position_text} "
                f"[Score: {factors['qualifying']:.1f}]"
            )
        
        # Circuit history
        if circuit_history and 'circuit' in factors:
            driver_history = [
                r for r in circuit_history
                if r.driver.driver_id == driver.driver_id
            ]
            
            if driver_history:
                wins = sum(1 for r in driver_history if r.position == 1)
                podiums = sum(1 for r in driver_history if r.position <= 3)
                
                if wins > 0:
                    reasoning.append(
                        f"Circuit History: {wins} win(s) at this circuit "
                        f"[Score: {factors['circuit']:.1f}]"
                    )
                elif podiums > 0:
                    reasoning.append(
                        f"Circuit History: {podiums} podium(s) at this circuit "
                        f"[Score: {factors['circuit']:.1f}]"
                    )
                else:
                    reasoning.append(
                        f"Circuit History: {len(driver_history)} race(s) at this circuit "
                        f"[Score: {factors['circuit']:.1f}]"
                    )
        
        return reasoning

    def analyze(
        self,
        race: Race,
        driver_standings: List[DriverStanding],
        constructor_standings: List[ConstructorStanding],
        recent_results: List[RaceResult],
        qualifying_results: Optional[List[QualifyingResult]] = None,
        circuit_history: Optional[List[RaceResult]] = None,
        top_n: int = 3
    ) -> List[DriverPrediction]:
        """
        Generate race winner predictions by analyzing all available data.
        
        Orchestrates all scoring calculations for each driver, handles missing
        data gracefully, and returns top N predictions sorted by confidence.
        
        Args:
            race: Race to predict
            driver_standings: Current driver championship standings
            constructor_standings: Current constructor championship standings
            recent_results: Recent race results for form calculation
            qualifying_results: Qualifying results for this race (optional)
            circuit_history: Historical results at this circuit (optional)
            top_n: Number of top predictions to return (default: 3)
            
        Returns:
            List of DriverPrediction objects sorted by confidence (highest first)
        """
        # Validate inputs
        if not race:
            logger.error("No race information provided")
            return []
        
        if not driver_standings:
            logger.error("No driver standings available for prediction")
            return []
        
        if not constructor_standings:
            logger.warning("No constructor standings available, predictions may be less accurate")
            constructor_standings = []
        
        if not recent_results:
            logger.warning("No recent results available, form calculations will use defaults")
            recent_results = []
        
        predictions = []
        
        # Track data completeness
        total_factors = 5
        available_factors = 3  # Championship, form, and team are always available
        if qualifying_results:
            available_factors += 1
        if circuit_history:
            available_factors += 1
        
        data_completeness = available_factors / total_factors
        
        logger.info(f"Analyzing race: {race.race_name}")
        logger.info(f"Data completeness: {data_completeness:.1%}")
        
        # Generate predictions for each driver in standings
        for driver_standing in driver_standings:
            try:
                driver = driver_standing.driver
                constructor = driver_standing.constructor
                
                # Validate driver and constructor data
                if not driver or not driver.driver_id:
                    logger.warning("Invalid driver data in standings, skipping")
                    continue
                
                if not constructor or not constructor.constructor_id:
                    logger.warning(f"Invalid constructor data for {driver.surname}, skipping")
                    continue
                
                # Calculate individual factor scores
                factors = {}
                
                # 1. Championship score (always available)
                try:
                    championship_score = self.calculate_championship_score(
                        driver, driver_standings
                    )
                    factors['championship'] = championship_score
                except Exception as e:
                    logger.warning(f"Failed to calculate championship score for {driver.surname}: {e}")
                    factors['championship'] = 50.0
                
                # 2. Recent form score (always available)
                try:
                    form_score = self.calculate_driver_form(driver, recent_results)
                    factors['form'] = form_score
                except Exception as e:
                    logger.warning(f"Failed to calculate form score for {driver.surname}: {e}")
                    factors['form'] = 50.0
                
                # 3. Team performance score (always available)
                try:
                    team_score = self.calculate_team_performance(
                        constructor, constructor_standings
                    )
                    factors['team'] = team_score
                except Exception as e:
                    logger.warning(f"Failed to calculate team score for {constructor.name}: {e}")
                    factors['team'] = 50.0
                
                # 4. Qualifying score (if available)
                qualifying_position = None
                if qualifying_results:
                    # Find driver's qualifying position
                    for qual_result in qualifying_results:
                        if qual_result.driver.driver_id == driver.driver_id:
                            qualifying_position = qual_result.position
                            try:
                                qualifying_score = self.calculate_qualifying_impact(
                                    qualifying_position
                                )
                                factors['qualifying'] = qualifying_score
                            except Exception as e:
                                logger.warning(f"Failed to calculate qualifying score: {e}")
                                factors['qualifying'] = 50.0
                            break
                    
                    if qualifying_position is None:
                        # Driver not in qualifying results, use neutral score
                        logger.debug(f"No qualifying data for {driver.surname}")
                        factors['qualifying'] = 50.0
                
                # 5. Circuit advantage score (if available)
                if circuit_history:
                    try:
                        circuit_score = self.calculate_circuit_advantage(
                            driver, race.circuit.circuit_id, circuit_history
                        )
                        factors['circuit'] = circuit_score
                    except Exception as e:
                        logger.warning(f"Failed to calculate circuit score: {e}")
                        factors['circuit'] = 50.0
                
                # Combine factors into final score
                try:
                    combined_score = self.combine_factors(factors)
                except Exception as e:
                    logger.warning(f"Failed to combine factors for {driver.surname}: {e}")
                    combined_score = 50.0
                
                # Calculate confidence adjusted for data completeness
                try:
                    confidence = self.calculate_confidence(combined_score, data_completeness)
                except Exception as e:
                    logger.warning(f"Failed to calculate confidence: {e}")
                    confidence = combined_score * data_completeness
                
                # Generate reasoning
                try:
                    reasoning = self.generate_reasoning(
                        driver=driver,
                        constructor=constructor,
                        factors=factors,
                        driver_standing=driver_standing,
                        qualifying_position=qualifying_position,
                        recent_results=recent_results,
                        circuit_history=circuit_history
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate reasoning: {e}")
                    reasoning = [f"Confidence: {confidence:.1f}%"]
                
                # Create prediction
                prediction = DriverPrediction(
                    driver=driver,
                    constructor=constructor,
                    confidence=confidence,
                    factors=factors,
                    reasoning=reasoning
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Failed to generate prediction for driver: {e}")
                continue
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        
        # Return top N predictions
        top_predictions = predictions[:top_n]
        
        logger.info(f"Generated {len(predictions)} predictions, returning top {top_n}")
        for i, pred in enumerate(top_predictions, 1):
            logger.info(
                f"{i}. {pred.driver.surname} ({pred.constructor.name}): "
                f"{pred.confidence:.1f}% confidence"
            )
        
        return top_predictions
