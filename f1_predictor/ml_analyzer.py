"""
Machine Learning-based prediction analyzer.

Uses trained Random Forest model to predict race winners instead of fixed weights.
Falls back to statistical analysis if model is not available.
"""

import logging
import pickle
import os
from typing import List, Dict, Optional
import numpy as np

from f1_predictor.models import (
    Race, Driver, Constructor, RaceResult, QualifyingResult,
    DriverStanding, ConstructorStanding, DriverPrediction
)
from f1_predictor.analyzer import PredictionAnalyzer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPredictionAnalyzer(PredictionAnalyzer):
    """
    Machine Learning-based analyzer that extends the statistical analyzer.
    
    Uses a trained Random Forest model to predict race winners. If the model
    is not available, falls back to the statistical method from parent class.
    
    The ML model learns optimal feature weights from historical data instead
    of using fixed weights.
    """
    
    def __init__(self, model_path: str = 'models/f1_predictor.pkl'):
        """
        Initialize ML analyzer.
        
        Args:
            model_path: Path to trained model file
        """
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained ML model from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(
                f"ML model not found at {self.model_path}. "
                "Run train_model.py to train a model, or use statistical mode."
            )
            return
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_loaded = True
            logger.info(f"ML model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            logger.warning("Falling back to statistical analysis")
    
    def _extract_features_for_driver(
        self,
        driver: Driver,
        constructor: Constructor,
        driver_standing: DriverStanding,
        driver_standings: List[DriverStanding],
        constructor_standings: List[ConstructorStanding],
        recent_results: List[RaceResult],
        qualifying_position: Optional[int],
        circuit_history: Optional[List[RaceResult]],
        race: Race
    ) -> np.ndarray:
        """
        Extract feature vector for a single driver.
        
        Uses the same feature calculations as the statistical model,
        but returns them as a numpy array for ML prediction.
        
        Args:
            driver: Driver to extract features for
            constructor: Driver's constructor
            driver_standing: Driver's championship standing
            driver_standings: All driver standings
            constructor_standings: All constructor standings
            recent_results: Recent race results
            qualifying_position: Qualifying position (if available)
            circuit_history: Historical results at circuit (if available)
            race: Race being predicted
            
        Returns:
            Numpy array of features [championship, form, team, qualifying, circuit]
        """
        # Calculate same features as statistical model
        championship_score = self.calculate_championship_score(driver, driver_standings)
        form_score = self.calculate_driver_form(driver, recent_results)
        team_score = self.calculate_team_performance(constructor, constructor_standings)
        
        # Qualifying score
        if qualifying_position:
            qualifying_score = self.calculate_qualifying_impact(qualifying_position)
        else:
            qualifying_score = 50.0  # Neutral if not available
        
        # Circuit history score
        if circuit_history:
            circuit_score = self.calculate_circuit_advantage(
                driver, race.circuit.circuit_id, circuit_history
            )
        else:
            circuit_score = 50.0  # Neutral if not available
        
        # Return as numpy array
        features = np.array([
            championship_score,
            form_score,
            team_score,
            qualifying_score,
            circuit_score
        ])
        
        return features
    
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
        Generate race winner predictions using ML model.
        
        If ML model is not available, falls back to statistical analysis.
        
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
        # Fall back to statistical if model not loaded
        if not self.model_loaded:
            logger.info("Using statistical analysis (ML model not available)")
            return super().analyze(
                race, driver_standings, constructor_standings,
                recent_results, qualifying_results, circuit_history, top_n
            )
        
        # Validate inputs
        if not race or not driver_standings:
            logger.error("Invalid input data for prediction")
            return []
        
        logger.info(f"Analyzing race with ML model: {race.race_name}")
        
        # Track data completeness
        total_factors = 5
        available_factors = 3  # Championship, form, and team always available
        if qualifying_results:
            available_factors += 1
        if circuit_history:
            available_factors += 1
        
        data_completeness = available_factors / total_factors
        logger.info(f"Data completeness: {data_completeness:.1%}")
        
        predictions = []
        
        # Extract features and predict for each driver
        for driver_standing in driver_standings:
            try:
                driver = driver_standing.driver
                constructor = driver_standing.constructor
                
                # Validate driver data
                if not driver or not driver.driver_id:
                    continue
                if not constructor or not constructor.constructor_id:
                    continue
                
                # Find qualifying position if available
                qualifying_position = None
                if qualifying_results:
                    for qual_result in qualifying_results:
                        if qual_result.driver.driver_id == driver.driver_id:
                            qualifying_position = qual_result.position
                            break
                
                # Extract features
                features = self._extract_features_for_driver(
                    driver=driver,
                    constructor=constructor,
                    driver_standing=driver_standing,
                    driver_standings=driver_standings,
                    constructor_standings=constructor_standings,
                    recent_results=recent_results,
                    qualifying_position=qualifying_position,
                    circuit_history=circuit_history,
                    race=race
                )
                
                # Predict using ML model
                # Reshape for single prediction: (1, n_features)
                features_reshaped = features.reshape(1, -1)
                
                # Get probability prediction
                # predict_proba returns [[prob_class_0, prob_class_1]]
                # We want the probability of winning (class 1)
                prediction_proba = self.model.predict_proba(features_reshaped)
                win_probability = prediction_proba[0][1]  # Probability of class 1 (win)
                
                # Convert to confidence score (0-100)
                confidence = win_probability * 100
                
                # Adjust for data completeness
                confidence = confidence * data_completeness
                
                # Create factors dict for reasoning (same as statistical)
                factors = {
                    'championship': features[0],
                    'form': features[1],
                    'team': features[2],
                    'qualifying': features[3],
                    'circuit': features[4]
                }
                
                # Generate reasoning
                reasoning = self.generate_reasoning(
                    driver=driver,
                    constructor=constructor,
                    factors=factors,
                    driver_standing=driver_standing,
                    qualifying_position=qualifying_position,
                    recent_results=recent_results,
                    circuit_history=circuit_history
                )
                
                # Add ML-specific note
                reasoning.insert(0, f"ML Model Confidence: {confidence:.1f}%")
                
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
                logger.error(f"Failed to generate ML prediction for driver: {e}")
                continue
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        
        # Return top N predictions
        top_predictions = predictions[:top_n]
        
        logger.info(f"Generated {len(predictions)} ML predictions, returning top {top_n}")
        for i, pred in enumerate(top_predictions, 1):
            logger.info(
                f"{i}. {pred.driver.surname} ({pred.constructor.name}): "
                f"{pred.confidence:.1f}% confidence"
            )
        
        return top_predictions
