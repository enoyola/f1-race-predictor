"""Data models for F1 Race Predictor."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# Race statuses that count as a classified finish. Jolpica reports lapped cars as
# "Lapped"; the original Ergast data used "+1 Lap", "+2 Laps", and so on.
FINISHED_STATUSES = ("Finished", "Lapped")


def is_finished(status: str) -> bool:
    """Return True if a result status represents a classified finish."""
    if not status:
        return False
    return status in FINISHED_STATUSES or status.startswith("+")


@dataclass
class Circuit:
    """Circuit information."""
    circuit_id: str
    circuit_name: str
    location: str
    country: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class Driver:
    """Driver information."""
    driver_id: str
    code: str  # e.g., "VER", "HAM"
    forename: str
    surname: str
    nationality: str

    @property
    def full_name(self) -> str:
        return f"{self.forename} {self.surname}"


@dataclass
class Constructor:
    """Constructor (team) information."""
    constructor_id: str
    name: str
    nationality: str


@dataclass
class Race:
    """Race event information."""
    season: int
    round: int
    race_name: str
    circuit: Circuit
    date: datetime
    is_sprint: bool = False
    total_rounds: Optional[int] = None

    @property
    def key(self) -> str:
        return f"{self.season}_{self.round}"


@dataclass
class RaceResult:
    """Result of a driver in a race (also used for sprint results)."""
    race: Race
    driver: Driver
    constructor: Constructor
    position: int
    points: float
    grid: int
    laps: int
    status: str

    @property
    def finished(self) -> bool:
        return is_finished(self.status)


@dataclass
class QualifyingResult:
    """Qualifying result for a driver."""
    race: Race
    driver: Driver
    constructor: Constructor
    position: int
    q1_time: Optional[str]
    q2_time: Optional[str]
    q3_time: Optional[str]


@dataclass
class DriverStanding:
    """Driver championship standing."""
    driver: Driver
    constructor: Constructor
    position: int
    points: float
    wins: int


@dataclass
class ConstructorStanding:
    """Constructor championship standing."""
    constructor: Constructor
    position: int
    points: float
    wins: int


@dataclass
class Weather:
    """Race-day weather at the circuit."""
    date: str
    precipitation_mm: Optional[float]
    precipitation_probability: Optional[float]  # percent, forecasts only
    temperature_max: Optional[float]
    source: str  # "archive" or "forecast"

    WET_THRESHOLD_MM = 1.0

    @property
    def is_wet(self) -> bool:
        if self.precipitation_mm is not None and self.precipitation_mm >= self.WET_THRESHOLD_MM:
            return True
        if self.precipitation_probability is not None and self.precipitation_probability >= 60:
            return True
        return False

    def describe(self) -> str:
        parts = []
        if self.precipitation_mm is not None:
            parts.append(f"{self.precipitation_mm:.1f} mm rain")
        if self.precipitation_probability is not None:
            parts.append(f"{self.precipitation_probability:.0f}% rain chance")
        if self.temperature_max is not None:
            parts.append(f"{self.temperature_max:.0f}°C")
        label = "wet" if self.is_wet else "dry"
        detail = ", ".join(parts)
        return f"{label} ({detail}, {self.source})" if detail else label


@dataclass
class Entry:
    """A driver and the constructor they race for at a given race."""
    driver: Driver
    constructor: Constructor


@dataclass
class RaceContext:
    """
    Everything known about a race *before* it is run.

    Built identically for live predictions, model training, and backtests so
    the features seen at training time match the ones seen at prediction time.
    """
    race: Race
    entries: List[Entry]
    driver_standings: List[DriverStanding]
    constructor_standings: List[ConstructorStanding]
    recent_results: List[RaceResult]          # all prior results (analyzers take the last N per driver)
    qualifying_results: List[QualifyingResult]
    sprint_results: List[RaceResult]          # this weekend's sprint, if any
    circuit_history: List[RaceResult]         # prior races at this circuit
    weather: Optional[Weather]
    rounds_completed: int                     # races already run this season
    standings_from_previous_season: bool = False
    actual_results: Optional[List[RaceResult]] = None  # populated for backtests
    notes: List[str] = field(default_factory=list)

    @property
    def has_qualifying(self) -> bool:
        return bool(self.qualifying_results)

    def qualifying_position(self, driver_id: str) -> Optional[int]:
        for q in self.qualifying_results:
            if q.driver.driver_id == driver_id:
                return q.position
        return None

    def sprint_position(self, driver_id: str) -> Optional[int]:
        for s in self.sprint_results:
            if s.driver.driver_id == driver_id:
                return s.position
        return None


@dataclass
class DriverPrediction:
    """Prediction for a single driver."""
    driver: Driver
    constructor: Constructor
    win_probability: float      # 0-1, sums to 1 across the field
    podium_probability: float   # 0-1, sums to 3 across the field
    score: float                # 0-100 raw model score
    factors: Dict[str, float]   # individual factor scores (0-100)
    reasoning: List[str]        # human-readable explanations
    grid_position: Optional[int] = None

    @property
    def confidence(self) -> float:
        """Win probability as a percentage (kept for backward compatibility)."""
        return self.win_probability * 100


@dataclass
class PredictionResult:
    """Complete prediction result for a race."""
    race: Race
    predictions: List[DriverPrediction]
    generated_at: datetime
    data_sources: List[str]
    data_completeness: float  # 0-1
    model_name: str = "statistical"
    weather: Optional[Weather] = None
    qualifying_available: bool = False
    notes: List[str] = field(default_factory=list)
    actual_results: Optional[List[RaceResult]] = None
    analysis: Optional[str] = None            # written analysis (AI analyst / verdict)
    components: Optional[Dict[str, Dict[str, float]]] = None  # verdict: driver_id -> model -> win prob


@dataclass
class PredictionError(Exception):
    """Prediction error with context."""
    error_type: str
    message: str
    suggestions: List[str]
    recoverable: bool

    def __str__(self) -> str:
        return self.message
