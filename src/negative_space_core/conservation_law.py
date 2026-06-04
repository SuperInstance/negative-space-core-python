"""ConservationLaw — verify avoidance ratio is conserved across scales."""

import math
from dataclasses import dataclass, field


@dataclass
class ConservationLaw:
    """Verify that avoidance ratio is conserved across population scales."""
    threshold: float = 0.01
    results: dict[int, tuple[float, float]] = field(default_factory=dict)

    def test_scale(self, pop_size: int, avoid_ratios: list[float]) -> bool:
        """Test conservation at a given scale. Returns True if std < threshold."""
        if not avoid_ratios:
            self.results[pop_size] = (0.0, 0.0)
            return True
        mean = sum(avoid_ratios) / len(avoid_ratios)
        variance = sum((r - mean) ** 2 for r in avoid_ratios) / len(avoid_ratios)
        std = math.sqrt(variance)
        self.results[pop_size] = (mean, std)
        return std < self.threshold

    def test_all_scales(self, data: dict[int, list[float]]) -> dict[int, bool]:
        """Test conservation at multiple scales. Returns {pop_size: passed}."""
        results = {}
        for pop_size, ratios in data.items():
            results[pop_size] = self.test_scale(pop_size, ratios)
        return results

    def all_conserved(self) -> bool:
        """Check if conservation holds at all tested scales."""
        return all(std < self.threshold for _, (_, std) in self.results.items())

    def report(self) -> str:
        lines = ["Conservation Law Verification", "=" * 40]
        for pop_size in sorted(self.results.keys()):
            mean, std = self.results[pop_size]
            status = "✓ PASS" if std < self.threshold else "✗ FAIL"
            lines.append(f"  N={pop_size:>5}: mean={mean:.4f}, std={std:.6f} {status}")
        lines.append(f"\nOverall: {'CONSERVED' if self.all_conserved() else 'VIOLATED'}")
        return "\n".join(lines)
