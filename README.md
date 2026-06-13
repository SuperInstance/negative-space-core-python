# Negative Space Core (Python)

**Negative Space Intelligence Theory** models intelligence through the lens of *avoidance behavior* — the insight that what an agent systematically avoids reveals as much (or more) about its knowledge as what it actively chooses. This Python library provides five composable primitives: `AvoidanceTracker`, `ConservationLaw`, `InferenceEngine`, `FeedbackLoop`, and `BatchAnalyzer`.

## Why It Matters

Standard ML pipelines optimize for positive labels: click-through rates, correct classifications, completed actions. But avoidance — the deliberate *not-doing* — carries high-information signals that positive-only models discard. A chess engine that never moves its queen to certain squares knows something about positional danger. A consumer who never enters certain stores knows something about pricing or preference. By formalizing avoidance as a first-class measurement, this library enables: anomaly detection (unusual avoidance patterns flag novel threats), knowledge inference (gaps in action space map to regions of latent expertise), and conservation testing (if avoidance ratios are scale-invariant, the underlying intelligence is structured, not random).

## How It Works

### AvoidanceTracker

Records `(action_type, confidence, timestamp)` tuples where `action_type ∈ {AVOID, CHOOSE, UNKNOWN}`. Computes running ratios:

```
R_avoid = Σ 1[a_i = AVOID] / N
R_choose = Σ 1[a_i = CHOOSE] / N
```

Standard deviation σ(actions) measures dispersion — low σ means consistent strategy, high σ means volatile. All ratios are O(N) computed on demand. Time complexity: O(1) per insertion.

### ConservationLaw

Partitions the action history into `k` temporal windows and tests whether the avoidance ratio is invariant:

```
H₀: R_avoid is constant across all k windows
Statistic: Var(R₁, ..., Rₖ) < τ (user threshold)
```

If variance < τ, the avoidance behavior is **conserved** — analogous to how physical quantities (energy, momentum) are conserved across scales in systems with symmetry. This is the library's central empirical claim: genuine intelligence produces scale-invariant avoidance ratios; random behavior does not.

### InferenceEngine

Scans the action space for contiguous regions of systematic avoidance ("gaps"). For each gap of width `w` and density `d`:

```
knowledge_estimate(w, d) = 1 - exp(-λ · w · d)
```

The total inferred knowledge aggregates all gaps. Wide, dense gaps imply strong latent knowledge about that region of the action space.

### FeedbackLoop

Closes the predict-observe-correct cycle. For each action, compare predicted avoidance probability `p̂` against actual outcome `y`:

```
error = y - p̂
RMSE = √(Σ error² / N)
model_adjustment = -α · mean(error)
```

When `mean_error` exceeds a threshold, the system recommends exploration — the model's predictions have diverged from reality.

### BatchAnalyzer

Processes multiple action streams in parallel, computing conservation statistics across batches. Useful for fleet-wide analysis of multiple agents.

## Quick Start

```python
from negative_space_core import AvoidanceTracker, ConservationLaw, InferenceEngine

tracker = AvoidanceTracker()

# Record ternary actions
for i in range(20):
    tracker.record(action_id=i, action="avoid", confidence=0.9, timestamp=float(i))
tracker.record(action_id=20, action="choose", confidence=0.8, timestamp=20.0)
for i in range(21, 30):
    tracker.record(action_id=i, action="avoid", confidence=0.85, timestamp=float(i))

print(f"Avoid ratio: {tracker.avoid_ratio():.3f}")  # ~0.93
print(f"Standard deviation: {tracker.std():.3f}")

# Check conservation law across 5 scales
conservation = ConservationLaw()
result = conservation.check(tracker, num_scales=5, threshold=0.05)
print(f"Conserved: {result.conserved}  (variance: {result.variance:.6f})")

# Infer knowledge from gaps
inference = InferenceEngine()
gaps = inference.find_gaps(tracker, min_gap_width=2.0)
print(f"Gaps: {gaps.num_gaps}, knowledge: {gaps.total_inferred_knowledge:.3f}")
```

Install: `pip install -e src/`

## API

| Class | Key Methods | Description |
|-------|------------|-------------|
| `AvoidanceTracker` | `record()`, `avoid_ratio()`, `choose_ratio()`, `std()` | Tracks ternary action stream |
| `ConservationLaw` | `check(tracker, num_scales, threshold)` | Tests scale-invariance of avoidance |
| `InferenceEngine` | `find_gaps(tracker, min_gap_width)` | Identifies knowledge gaps |
| `FeedbackLoop` | `record()`, `update()`, `should_explore()` | Prediction-error feedback |
| `BatchAnalyzer` | `analyze(trackers[])` | Multi-stream batch analysis |

## Architecture Notes

This is the Python counterpart to `negative-space-core-c`, implementing the **η (eta)** component of SuperInstance's γ + η = C framework. Where γ tracks constructive intelligence (what the agent builds, chooses, generates), η tracks the subtractive intelligence (what the agent avoids, rejects, leaves empty). Their sum yields **C (Competence)** — the total capacity measure. The Python implementation prioritizes readability and composability over the C version's raw speed. See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Gibson, J. J. (1979). *The Ecological Approach to Visual Perception*. — Affordances and what the environment offers for avoidance.
2. Noether, E. (1918). "Invariante Variationsprobleme." *Nachr. d. König. Gesellsch. d. Wiss. zu Göttingen*, 235–257. — The mathematical foundation of conservation laws and symmetry.
3. Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11, 127–138. — Avoidance as surprise minimization.

## License

MIT
