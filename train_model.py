"""
Train ML model for F1 race prediction.

This script fetches historical F1 data, extracts features, and trains
a Random Forest classifier to predict race winners.

Run this script once to create the model:
    python train_model.py

The trained model will be saved to models/f1_predictor.pkl
"""

import os
import pickle
import logging
from datetime import datetime
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from f1_predictor.data_fetcher import F1DataFetcher
from f1_predictor.analyzer import PredictionAnalyzer
from f1_predictor.models import RaceResult
from f1_predictor.cache import DataCache


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains ML model on historical F1 data."""
    
    def __init__(self, years_back: int = 5):
        """
        Initialize trainer.
        
        Args:
            years_back: Number of years of historical data to use
        """
        self.years_back = years_back
        cache = DataCache(".f1_cache")
        self.fetcher = F1DataFetcher(cache=cache, use_cache=True)
        self.analyzer = PredictionAnalyzer()
        self.current_year = datetime.now().year
        
    def fetch_historical_data(self) -> List[RaceResult]:
        """
        Fetch historical race results for training.
        
        Returns:
            List of race results from past years
        """
        logger.info(f"Fetching {self.years_back} years of historical data...")
        
        all_results = []
        start_year = self.current_year - self.years_back
        
        for year in range(start_year, self.current_year):
            try:
                logger.info(f"Fetching season {year}...")
                season_results = self.fetcher.get_current_season_results(year)
                all_results.extend(season_results)
                logger.info(f"  Loaded {len(season_results)} results from {year}")
            except Exception as e:
                logger.warning(f"Failed to fetch {year} season: {e}")
                continue
        
        logger.info(f"Total historical results: {len(all_results)}")
        return all_results
    
    def extract_features_and_labels(
        self,
        results: List[RaceResult]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from race results.
        
        For each race result, we extract:
        - Features: [championship_score, form_score, team_score, qualifying_score, circuit_score]
        - Label: 1 if driver won, 0 otherwise
        
        Args:
            results: List of race results
            
        Returns:
            Tuple of (features array, labels array)
        """
        logger.info("Extracting features and labels...")
        
        X = []  # Features
        y = []  # Labels (1 = won, 0 = didn't win)
        
        # Group results by race
        races_dict = {}
        for result in results:
            race_key = f"{result.race.season}_{result.race.round}"
            if race_key not in races_dict:
                races_dict[race_key] = []
            races_dict[race_key].append(result)
        
        logger.info(f"Processing {len(races_dict)} races...")
        
        processed_count = 0
        for race_key, race_results in races_dict.items():
            try:
                # Sort by position to identify winner
                race_results.sort(key=lambda r: r.position)
                
                if not race_results:
                    continue
                
                winner = race_results[0]  # Position 1
                
                # For each driver in the race, extract features
                for result in race_results:
                    # Skip if missing critical data
                    if not result.driver or not result.constructor:
                        continue
                    
                    # Extract features using analyzer methods
                    # Note: In real training, we'd need historical standings/form
                    # For simplicity, we'll use position-based proxies
                    
                    # Championship score proxy: inverse of position
                    championship_score = max(0, 100 - (result.position * 5))
                    
                    # Form score proxy: based on grid position
                    form_score = max(0, 100 - (result.grid * 5))
                    
                    # Team score proxy: based on constructor
                    # (In real implementation, use actual constructor standings)
                    team_score = 50.0  # Neutral for now
                    
                    # Qualifying score: based on grid position
                    qualifying_score = self.analyzer.calculate_qualifying_impact(
                        result.grid if result.grid > 0 else 20
                    )
                    
                    # Circuit score: neutral (would need historical circuit data)
                    circuit_score = 50.0
                    
                    # Create feature vector
                    features = [
                        championship_score,
                        form_score,
                        team_score,
                        qualifying_score,
                        circuit_score
                    ]
                    
                    # Label: 1 if this driver won, 0 otherwise
                    label = 1 if result.driver.driver_id == winner.driver.driver_id else 0
                    
                    X.append(features)
                    y.append(label)
                
                processed_count += 1
                if processed_count % 50 == 0:
                    logger.info(f"  Processed {processed_count}/{len(races_dict)} races...")
                    
            except Exception as e:
                logger.warning(f"Failed to process race {race_key}: {e}")
                continue
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        logger.info(f"Extracted {len(X_array)} samples")
        logger.info(f"  Winners: {sum(y_array)} ({sum(y_array)/len(y_array)*100:.1f}%)")
        logger.info(f"  Non-winners: {len(y_array) - sum(y_array)}")
        
        return X_array, y_array
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',  # Handle imbalanced data
            n_jobs=-1  # Use all CPU cores
        )
        
        logger.info("Fitting model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        logger.info("\nModel Evaluation:")
        logger.info("=" * 60)
        
        # Training accuracy
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        logger.info(f"Training Accuracy: {train_acc:.3f}")
        
        # Test accuracy
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        logger.info(f"Test Accuracy: {test_acc:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        logger.info(f"Cross-validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Feature importance
        logger.info("\nFeature Importance:")
        feature_names = ['Championship', 'Form', 'Team', 'Qualifying', 'Circuit']
        for name, importance in zip(feature_names, model.feature_importances_):
            logger.info(f"  {name}: {importance:.3f}")
        
        # Classification report
        logger.info("\nClassification Report (Test Set):")
        logger.info("\n" + classification_report(y_test, test_pred, target_names=['No Win', 'Win']))
        
        return model
    
    def save_model(self, model: RandomForestClassifier, path: str = 'models/f1_predictor.pkl'):
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            path: Path to save model
        """
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        logger.info(f"Saving model to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info("Model saved successfully!")
    
    def run(self):
        """Run the complete training pipeline."""
        logger.info("=" * 60)
        logger.info("F1 Race Predictor - ML Model Training")
        logger.info("=" * 60)
        
        # Step 1: Fetch data
        results = self.fetch_historical_data()
        
        if len(results) < 100:
            logger.error("Not enough historical data to train model!")
            logger.error("Need at least 100 race results, but got {len(results)}")
            return
        
        # Step 2: Extract features
        X, y = self.extract_features_and_labels(results)
        
        if len(X) < 100:
            logger.error("Not enough training samples!")
            return
        
        # Step 3: Train model
        model = self.train_model(X, y)
        
        # Step 4: Save model
        self.save_model(model)
        
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("You can now use the ML model with:")
        logger.info("  python -m f1_predictor.cli --ml")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    trainer = ModelTrainer(years_back=5)
    trainer.run()


if __name__ == '__main__':
    main()
