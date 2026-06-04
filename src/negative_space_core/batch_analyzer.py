"""BatchAnalyzer — analyze batches of agent decisions."""

import math
from dataclasses import dataclass, field


@dataclass
class BatchAnalyzer:
    """Analyzes batches of ternary agent decisions."""
    batches: list[list[int]] = field(default_factory=list)

    def add_batch(self, actions: list[int]) -> dict[str, float]:
        """Add a batch of actions and return statistics."""
        self.batches.append(actions)
        return self.analyze(actions)

    @staticmethod
    def analyze(actions: list[int]) -> dict[str, float]:
        """Analyze a single batch."""
        n = len(actions)
        if n == 0:
            return {"avoid": 0, "unknown": 0, "choose": 0, "entropy": 0, "active_rate": 0}

        avoid = sum(1 for a in actions if a == -1)
        unknown = sum(1 for a in actions if a == 0)
        choose = sum(1 for a in actions if a == 1)

        # Shannon entropy
        entropy = 0.0
        for count in [avoid, unknown, choose]:
            if count > 0:
                p = count / n
                entropy -= p * math.log2(p)

        return {
            "avoid": avoid / n,
            "unknown": unknown / n,
            "choose": choose / n,
            "entropy": round(entropy, 4),
            "active_rate": choose / n,
        }

    def population_summary(self) -> dict[str, float]:
        """Summary across all batches."""
        if not self.batches:
            return {}
        all_stats = [self.analyze(b) for b in self.batches]
        keys = all_stats[0].keys()
        return {k: sum(s[k] for s in all_stats) / len(all_stats) for k in keys}

    def avoid_choose_ratio(self) -> float:
        """Compute overall avoid:choose ratio."""
        total_avoid = sum(sum(1 for a in b if a == -1) for b in self.batches)
        total_choose = sum(sum(1 for a in b if a == 1) for b in self.batches)
        if total_choose == 0:
            return 0.0
        return total_avoid / total_choose

    def conservation_std(self) -> float:
        """Std of avoid ratio across batches (conservation metric)."""
        if len(self.batches) < 2:
            return 0.0
        ratios = [sum(1 for a in b if a == -1) / len(b) for b in self.batches if b]
        mean = sum(ratios) / len(ratios)
        variance = sum((r - mean) ** 2 for r in ratios) / len(ratios)
        return math.sqrt(variance)
