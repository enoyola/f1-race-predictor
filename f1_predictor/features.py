"""
Feature engineering shared by the statistical analyzer, the ML model, the
training script and the backtester.

Every function here works from a RaceContext, which only contains information
available before the race starts. That keeps training features identical to
the features seen at prediction time.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from f1_predictor.models import (
    ConstructorStanding, DriverStanding, Entry, RaceContext, RaceResult
)

RECENT_RACES_COUNT = 5
RELIABILITY_WINDOW = 10
TEAMMATE_WINDOW = 10

# Order matters: this is the ML feature vector layout, stored with the model.
FEATURE_NAMES = [
    "championship",
    "form",
    "team",
    "qualifying",
    "circuit",
    "reliability",
    "teammate",
    "sprint",
    "grid_position",
    "has_qualifying",
    "season_progress",
    "avg_finish_last5",
    "points_share",
    "team_points_share",
    "wet",
    "weather_known",
    "is_sprint_weekend",
]

# The subset of features that are 0-100 "factor" scores shown to users.
FACTOR_NAMES = ["championship", "form", "team", "qualifying", "circuit", "reliability", "teammate", "sprint"]

FACTOR_LABELS = {
    "championship": "Championship Position",
    "form": "Recent Form",
    "team": "Team Performance",
    "qualifying": "Qualifying Position",
    "circuit": "Circuit History",
    "reliability": "Reliability",
    "teammate": "Teammate Comparison",
    "sprint": "Sprint Result",
}

NEUTRAL = 50.0


# ----------------------------------------------------------------------
# Individual scoring functions (0-100)
# ----------------------------------------------------------------------

def driver_recent_results(driver_id: str, results: List[RaceResult], n: int) -> List[RaceResult]:
    """The driver's last n results, most recent first."""
    mine = [r for r in results if r.driver.driver_id == driver_id]
    mine.sort(key=lambda r: r.race.date, reverse=True)
    return mine[:n]


def form_score(driver_id: str, recent_results: List[RaceResult], n: int = RECENT_RACES_COUNT) -> float:
    """
    Recent form from the last n races.

    Win 25, P2 18, P3 15, P4 12, P5 10, P6 8, P7 6, P8 4, P9 2, P10 1,
    outside the points 0, retirement -2. Normalized to 0-100.
    """
    mine = driver_recent_results(driver_id, recent_results, n)
    if not mine:
        return NEUTRAL

    points_table = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    total = 0.0
    for r in mine:
        if not r.finished:
            total += -2
        else:
            total += points_table.get(r.position, 0)

    races = len(mine)
    max_points = 25 * races
    min_points = -2 * races
    normalized = ((total - min_points) / (max_points - min_points)) * 100
    return max(0.0, min(100.0, normalized))


def avg_finish(driver_id: str, recent_results: List[RaceResult], n: int = RECENT_RACES_COUNT) -> float:
    mine = driver_recent_results(driver_id, recent_results, n)
    if not mine:
        return 12.0
    return sum(r.position for r in mine) / len(mine)


def team_score(constructor_id: str, standings: List[ConstructorStanding]) -> float:
    """Constructor standing position with a bonus for share of wins."""
    if not standings:
        return NEUTRAL
    standing = next((s for s in standings if s.constructor.constructor_id == constructor_id), None)
    if standing is None:
        return NEUTRAL
    num_teams = len(standings)
    position_score = max(0.0, 100 - ((standing.position - 1) * (100 / num_teams)))
    total_wins = sum(s.wins for s in standings)
    win_bonus = (standing.wins / total_wins) * 10 if total_wins > 0 else 0.0
    return min(100.0, position_score + win_bonus)


def team_points_share(constructor_id: str, standings: List[ConstructorStanding]) -> float:
    if not standings:
        return 0.0
    leader = max(s.points for s in standings)
    if leader <= 0:
        return 0.0
    standing = next((s for s in standings if s.constructor.constructor_id == constructor_id), None)
    return standing.points / leader if standing else 0.0


def circuit_score(driver_id: str, history: List[RaceResult]) -> float:
    """Average of win 30 / podium 15 / points 5 per past race at the circuit."""
    if not history:
        return NEUTRAL
    mine = [r for r in history if r.driver.driver_id == driver_id]
    if not mine:
        return NEUTRAL
    total = 0.0
    for r in mine:
        if r.position == 1:
            total += 30
        elif r.position <= 3:
            total += 15
        elif r.position <= 10:
            total += 5
    normalized = (total / len(mine)) / 30 * 100
    return max(0.0, min(100.0, normalized))


def position_score(position: Optional[int]) -> float:
    """Grid or sprint position mapped to 0-100 (pole 100, P2 90, ...)."""
    if position is None or position <= 0:
        return NEUTRAL
    if position == 1:
        return 100.0
    if position == 2:
        return 90.0
    if position == 3:
        return 80.0
    if position <= 5:
        return 70.0
    if position <= 10:
        return 50.0
    return float(max(0, 50 - ((position - 10) * 5)))


qualifying_score = position_score
sprint_score = position_score


def championship_score(driver_id: str, standings: List[DriverStanding]) -> float:
    """Driver standing position scaled by points gap to the leader."""
    if not standings:
        return NEUTRAL
    standing = next((s for s in standings if s.driver.driver_id == driver_id), None)
    if standing is None:
        return NEUTRAL
    num_drivers = len(standings)
    score = max(0.0, 100 - ((standing.position - 1) * (100 / num_drivers)))
    leader_points = standings[0].points
    if leader_points > 0:
        ratio = standing.points / leader_points
        score *= 0.5 + ratio * 0.5
    return max(0.0, min(100.0, score))


def points_share(driver_id: str, standings: List[DriverStanding]) -> float:
    if not standings:
        return 0.0
    leader_points = standings[0].points
    if leader_points <= 0:
        return 0.0
    standing = next((s for s in standings if s.driver.driver_id == driver_id), None)
    return standing.points / leader_points if standing else 0.0


def reliability_score(driver_id: str, recent_results: List[RaceResult], window: int = RELIABILITY_WINDOW) -> float:
    """100 minus the retirement rate over the last `window` races."""
    mine = driver_recent_results(driver_id, recent_results, window)
    if not mine:
        return 85.0  # roughly the field-wide average finish rate
    dnfs = sum(1 for r in mine if not r.finished)
    return 100.0 * (1 - dnfs / len(mine))


def dnf_rate(driver_id: str, recent_results: List[RaceResult], window: int = RELIABILITY_WINDOW) -> Optional[float]:
    mine = driver_recent_results(driver_id, recent_results, window)
    if not mine:
        return None
    return sum(1 for r in mine if not r.finished) / len(mine)


def teammate_delta(driver_id: str, constructor_id: str, recent_results: List[RaceResult], window: int = TEAMMATE_WINDOW) -> Optional[float]:
    """
    Average (teammate position - driver position) over recent shared races.
    Positive means the driver usually beats their teammate.
    """
    mine = driver_recent_results(driver_id, recent_results, window)
    if not mine:
        return None
    by_race: Dict[str, List[RaceResult]] = {}
    for r in recent_results:
        by_race.setdefault(r.race.key, []).append(r)
    deltas = []
    for r in mine:
        if r.constructor.constructor_id != constructor_id:
            continue
        teammates = [
            t for t in by_race.get(r.race.key, [])
            if t.constructor.constructor_id == constructor_id and t.driver.driver_id != driver_id
        ]
        if teammates:
            deltas.append(teammates[0].position - r.position)
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def teammate_score(driver_id: str, constructor_id: str, recent_results: List[RaceResult], window: int = TEAMMATE_WINDOW) -> float:
    delta = teammate_delta(driver_id, constructor_id, recent_results, window)
    if delta is None:
        return NEUTRAL
    return max(0.0, min(100.0, NEUTRAL + delta * 5))


# ----------------------------------------------------------------------
# Feature vectors
# ----------------------------------------------------------------------

@dataclass
class DriverFeatures:
    entry: Entry
    values: Dict[str, float]
    qualifying_position: Optional[int]
    sprint_position: Optional[int]

    def vector(self) -> List[float]:
        return [self.values[name] for name in FEATURE_NAMES]

    @property
    def factors(self) -> Dict[str, float]:
        """The 0-100 factor scores for display."""
        return {name: self.values[name] for name in FACTOR_NAMES if name in self.values}


def extract_features(context: RaceContext, entry: Entry) -> DriverFeatures:
    """Compute every feature for one driver in a race context."""
    driver_id = entry.driver.driver_id
    constructor_id = entry.constructor.constructor_id

    quali_pos = context.qualifying_position(driver_id)
    sprint_pos = context.sprint_position(driver_id)
    has_quali = context.has_qualifying
    total_rounds = context.race.total_rounds or 24
    season_progress = min(1.0, context.rounds_completed / max(1, total_rounds))
    if context.standings_from_previous_season:
        season_progress = 0.0

    weather = context.weather
    values: Dict[str, float] = {
        "championship": championship_score(driver_id, context.driver_standings),
        "form": form_score(driver_id, context.recent_results),
        "team": team_score(constructor_id, context.constructor_standings),
        "qualifying": qualifying_score(quali_pos) if has_quali else NEUTRAL,
        "circuit": circuit_score(driver_id, context.circuit_history),
        "reliability": reliability_score(driver_id, context.recent_results),
        "teammate": teammate_score(driver_id, constructor_id, context.recent_results),
        "sprint": sprint_score(sprint_pos) if sprint_pos else NEUTRAL,
        "grid_position": float(quali_pos) if (has_quali and quali_pos) else 21.0,
        "has_qualifying": 1.0 if has_quali else 0.0,
        "season_progress": season_progress,
        "avg_finish_last5": avg_finish(driver_id, context.recent_results),
        "points_share": points_share(driver_id, context.driver_standings),
        "team_points_share": team_points_share(constructor_id, context.constructor_standings),
        "wet": 1.0 if (weather is not None and weather.is_wet) else 0.0,
        "weather_known": 1.0 if weather is not None else 0.0,
        "is_sprint_weekend": 1.0 if context.race.is_sprint else 0.0,
    }
    return DriverFeatures(
        entry=entry,
        values=values,
        qualifying_position=quali_pos if has_quali else None,
        sprint_position=sprint_pos,
    )


def extract_all(context: RaceContext) -> List[DriverFeatures]:
    return [extract_features(context, entry) for entry in context.entries]
