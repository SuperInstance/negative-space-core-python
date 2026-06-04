"""InferenceEngine — deduce knowledge from negative spaces."""

from dataclasses import dataclass, field


@dataclass
class InferenceEngine:
    """Makes deductions from the gaps between avoidance regions."""
    avoidances: dict[int, float] = field(default_factory=dict)  # position -> avoid_count
    deductions: list[dict] = field(default_factory=list)

    def record_avoidance(self, position: int, count: float) -> None:
        self.avoidances[position] = count

    def find_gaps(self) -> list[tuple[int, int]]:
        """Find gaps (unexplored regions) between avoidance regions."""
        if len(self.avoidances) < 2:
            return []
        positions = sorted(self.avoidances.keys())
        gaps = []
        for i in range(len(positions) - 1):
            if positions[i + 1] - positions[i] > 1:
                gaps.append((positions[i], positions[i + 1]))
        return gaps

    def infer(self) -> list[dict]:
        """Deduce what the gaps between avoidances imply."""
        self.deductions.clear()
        gaps = self.find_gaps()
        for left, right in gaps:
            left_intensity = self.avoidances[left]
            right_intensity = self.avoidances[right]
            avg_intensity = (left_intensity + right_intensity) / 2
            gap_size = right - left - 1

            if avg_intensity > 0.7:
                inference = "interpolation"
                confidence = avg_intensity * 0.8
                meaning = f"High avoidance both sides → gap likely avoidable too"
            elif avg_intensity < 0.3:
                inference = "exclusion"
                confidence = (1.0 - avg_intensity) * 0.6
                meaning = f"Low avoidance both sides → gap may be worth exploring"
            else:
                inference = "boundary"
                confidence = 0.5
                meaning = f"Mixed avoidance → gap is a decision boundary"

            self.deductions.append({
                "left": left,
                "right": right,
                "gap_size": gap_size,
                "inference": inference,
                "confidence": round(confidence, 3),
                "meaning": meaning,
            })
        return self.deductions

    def high_confidence_deductions(self, min_confidence: float = 0.6) -> list[dict]:
        return [d for d in self.deductions if d["confidence"] >= min_confidence]
