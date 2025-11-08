"""Data models for F1 Race Predictor."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List



@dataclass
class Circuit:
    """Represents an F1 circuit."""
    circuit_id: str
    circuit_name: str
    location: str
    country: str


@dataclass
class Driver:
    """Represents an F1 driver."""
    driver_id: str
    code: str  # e.g., "VER", "HAM"
    forename: str
    surname: str
    nationality: str


@dataclass
class Constructor:
    """Represents an F1 constructor/team."""
    constructor_id: str
    name: str
    nationality: str


@dataclass
class Race:
    """Represents an F1 race."""
    season: int
    round: int
    race_name: str
    circuit: Circuit
    date: datetime


@dataclass
class RaceResult:
    """Represents a race result for a driver."""
    race: Race
    driver: Driver
    constructor: Constructor
    position: int
    points: float
    grid: int
    laps: int
    status: str


@dataclass
class QualifyingResult:
    """Represents a qualifying result for a driver."""
    race: Race
    driver: Driver
    constructor: Constructor
    position: int
    q1_time: Optional[str]
    q2_time: Optional[str]
    q3_time: Optional[str]


@dataclass
class DriverStanding:
    """Represents a driver's championship standing."""
    driver: Driver
    constructor: Constructor
    position: int
    points: float
    wins: int


@dataclass
class ConstructorStanding:
    """Represents a constructor's championship standing."""
    constructor: Constructor
    position: int
    points: float
    wins: int


@dataclass
class DriverPrediction:
    """Represents a prediction for a driver's race performance."""
    driver: Driver
    constructor: Constructor
    confidence: float  # 0-100
    factors: Dict[str, float]  # Individual factor scores
    reasoning: List[str]  # Human-readable explanations


@dataclass
class PredictionResult:
    """Represents the complete prediction result for a race."""
    race: Race
    predictions: List[DriverPrediction]
    generated_at: datetime
    data_sources: List[str]
    data_completeness: float  # 0-1


@dataclass
class PredictionError(Exception):
    """Represents an error that occurred during prediction."""
    error_type: str
    message: str
    suggestions: List[str]
    recoverable: bool
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        return self.message
