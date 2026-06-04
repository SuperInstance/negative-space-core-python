"""AvoidanceTracker — tracks per-position avoidance history and ratios."""

from dataclasses import dataclass, field


@dataclass
class AvoidanceTracker:
    """Tracks avoidance behavior across positions and generations."""
    positions: int
    history: list[list[float]] = field(default_factory=list)

    def record(self, actions: list[int]) -> dict[str, float]:
        """Record a generation's actions. Actions are -1 (avoid), 0 (unknown), +1 (choose)."""
        assert len(actions) == self.positions
        avoid = sum(1 for a in actions if a == -1)
        unknown = sum(1 for a in actions if a == 0)
        choose = sum(1 for a in actions if a == 1)
        n = len(actions)
        ratios = {
            "avoid": avoid / n,
            "unknown": unknown / n,
            "choose": choose / n,
        }
        self.history.append([ratios["avoid"], ratios["unknown"], ratios["choose"]])
        return ratios

    def avoid_ratio(self) -> float:
        """Mean avoidance ratio across all recorded generations."""
        if not self.history:
            return 0.0
        return sum(h[0] for h in self.history) / len(self.history)

    def choose_ratio(self) -> float:
        if not self.history:
            return 0.0
        return sum(h[2] for h in self.history) / len(self.history)

    def unknown_ratio(self) -> float:
        if not self.history:
            return 0.0
        return sum(h[1] for h in self.history) / len(self.history)

    def avoid_std(self) -> float:
        """Standard deviation of avoidance ratio across generations."""
        if not self.history:
            return 0.0
        mean = self.avoid_ratio()
        variance = sum((h[0] - mean) ** 2 for h in self.history) / len(self.history)
        return variance ** 0.5

    def generations(self) -> int:
        return len(self.history)
