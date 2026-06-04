"""FeedbackLoop — update avoidance model based on outcomes."""

from dataclasses import dataclass, field


@dataclass
class FeedbackLoop:
    """Updates avoidance model based on feedback. Implements balanced v5 learning."""
    decay: float = 0.9
    margin: float = 0.0
    forced_explore_interval: int = 5
    memory: dict[str, float] = field(default_factory=dict)
    predictions: list[dict] = field(default_factory=list)
    step: int = 0
    correct: int = 0
    total: int = 0

    def predict(self, option: str) -> int:
        """Predict action for an option: -1 (avoid), 0 (explore), +1 (choose)."""
        self.step += 1

        # Forced exploration
        if self.step % self.forced_explore_interval == 0:
            return 0

        if option not in self.memory:
            return 0  # unknown → explore

        score = self.memory[option]
        if score < self.margin:
            return -1  # avoid
        return 1  # choose

    def update(self, option: str, outcome: float, predicted: int) -> None:
        """Update memory with outcome. Track prediction accuracy."""
        # Exponential moving average
        if option in self.memory:
            self.memory[option] = self.decay * self.memory[option] + (1 - self.decay) * outcome
        else:
            self.memory[option] = outcome

        # Track accuracy
        actual = -1 if outcome < self.margin else 1
        self.total += 1
        if predicted == actual:
            self.correct += 1

        self.predictions.append({
            "option": option,
            "predicted": predicted,
            "outcome": outcome,
            "actual": actual,
        })

    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def run_round(self, options: dict[str, float]) -> dict[str, int]:
        """Run one round: predict + update for all options."""
        decisions = {}
        for option, reward in options.items():
            pred = self.predict(option)
            self.update(option, reward, pred)
            decisions[option] = pred
        return decisions
