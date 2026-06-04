# negative-space-core-python

Python implementation of negative space intelligence — the core theory that intelligence is what you learn to AVOID.

## The 5 Laws

1. Negative space discovers hidden structure (60% avoidance through feedback alone)
2. Avoidance dominates choice (294:1 ratio)
3. Strategy species coexist stably (100% resilience)
4. Population > Individual (+0.075 fitness advantage)
5. Avoidance ratio CONSERVED across scales (std=0.001 from 10 to 5000 agents)

## Install

```bash
pip install negative-space-core
```

## Usage

```python
from negative_space_core import AvoidanceTracker, ConservationLaw, BatchAnalyzer

# Track avoidance across generations
tracker = AvoidanceTracker(positions=100)
for _ in range(50):
    actions = simulate_population(100)
    tracker.record(actions)

print(f"Avoid ratio: {tracker.avoid_ratio():.3f} ± {tracker.avoid_std():.4f}")

# Verify conservation law
cl = ConservationLaw(threshold=0.02)
data = {10: [...], 100: [...], 1000: [...], 5000: [...]}
results = cl.test_all_scales(data)
print(cl.report())
```

## API

- **AvoidanceTracker** — ratio tracking, std computation, conservation verification
- **ConservationLaw** — multi-scale conservation testing with reports
- **InferenceEngine** — deduce knowledge from gaps between avoidances
- **FeedbackLoop** — balanced v5 learning with forced exploration and decay
- **BatchAnalyzer** — batch statistics, avoid:choose ratio, conservation std

## License

MIT
