"""Tests for negative-space-core-python."""
import math
from negative_space_core import (
    AvoidanceTracker,
    ConservationLaw,
    InferenceEngine,
    FeedbackLoop,
    BatchAnalyzer,
)


class TestAvoidanceTracker:
    def test_record(self):
        t = AvoidanceTracker(positions=10)
        actions = [-1]*5 + [0]*3 + [1]*2
        r = t.record(actions)
        assert abs(r["avoid"] - 0.5) < 0.01
        assert abs(r["unknown"] - 0.3) < 0.01
        assert abs(r["choose"] - 0.2) < 0.01

    def test_mean_ratios(self):
        t = AvoidanceTracker(positions=10)
        for _ in range(5):
            t.record([-1]*5 + [0]*3 + [1]*2)
        assert abs(t.avoid_ratio() - 0.5) < 0.01
        assert abs(t.choose_ratio() - 0.2) < 0.01

    def test_std(self):
        t = AvoidanceTracker(positions=10)
        for _ in range(5):
            t.record([-1]*5 + [0]*3 + [1]*2)
        assert t.avoid_std() < 0.01  # constant ratio

    def test_empty(self):
        t = AvoidanceTracker(positions=10)
        assert t.avoid_ratio() == 0.0
        assert t.generations() == 0

    def test_conservation_across_scales(self):
        for n in [10, 100, 1000, 5000]:
            t = AvoidanceTracker(positions=n)
            for _ in range(50):
                actions = []
                for i in range(n):
                    if i % 10 < 5: actions.append(-1)
                    elif i % 10 < 8: actions.append(0)
                    else: actions.append(1)
                t.record(actions)
            assert abs(t.avoid_ratio() - 0.5) < 0.01
            assert t.avoid_std() < 0.02


class TestConservationLaw:
    def test_conserved(self):
        cl = ConservationLaw(threshold=0.02)
        result = cl.test_scale(100, [0.5, 0.5, 0.5, 0.5, 0.5])
        assert result

    def test_violated(self):
        cl = ConservationLaw(threshold=0.01)
        result = cl.test_scale(100, [0.1, 0.3, 0.5, 0.7, 0.9])
        assert not result

    def test_all_scales(self):
        cl = ConservationLaw(threshold=0.02)
        data = {
            10: [0.5]*10,
            100: [0.5]*10,
            1000: [0.5]*10,
        }
        results = cl.test_all_scales(data)
        assert all(results.values())
        assert cl.all_conserved()

    def test_report(self):
        cl = ConservationLaw(threshold=0.02)
        cl.test_scale(100, [0.5]*10)
        report = cl.report()
        assert "PASS" in report


class TestInferenceEngine:
    def test_no_gaps(self):
        ie = InferenceEngine()
        ie.record_avoidance(0, 0.8)
        ie.record_avoidance(1, 0.7)
        gaps = ie.find_gaps()
        assert len(gaps) == 0

    def test_find_gaps(self):
        ie = InferenceEngine()
        ie.record_avoidance(0, 0.8)
        ie.record_avoidance(5, 0.7)
        gaps = ie.find_gaps()
        assert len(gaps) == 1
        assert gaps[0] == (0, 5)

    def test_infer_interpolation(self):
        ie = InferenceEngine()
        ie.record_avoidance(0, 0.9)
        ie.record_avoidance(5, 0.9)
        deductions = ie.infer()
        assert len(deductions) == 1
        assert deductions[0]["inference"] == "interpolation"
        assert deductions[0]["confidence"] > 0.6

    def test_infer_exclusion(self):
        ie = InferenceEngine()
        ie.record_avoidance(0, 0.1)
        ie.record_avoidance(5, 0.1)
        deductions = ie.infer()
        assert len(deductions) == 1
        assert deductions[0]["inference"] == "exclusion"

    def test_high_confidence_filter(self):
        ie = InferenceEngine()
        ie.record_avoidance(0, 0.95)
        ie.record_avoidance(10, 0.95)
        ie.infer()
        high = ie.high_confidence_deductions(0.5)
        assert len(high) >= 1


class TestFeedbackLoop:
    def test_explore_unknown(self):
        fl = FeedbackLoop()
        pred = fl.predict("new_option")
        assert pred == 0  # unknown → explore

    def test_choose_good(self):
        fl = FeedbackLoop()
        fl.memory["good"] = 1.0
        pred = fl.predict("good")
        assert pred == 1  # choose

    def test_avoid_bad(self):
        fl = FeedbackLoop()
        fl.memory["bad"] = -0.5
        pred = fl.predict("bad")
        assert pred == -1  # avoid

    def test_memory_decay(self):
        fl = FeedbackLoop(decay=0.5)
        fl.memory["opt"] = 1.0
        fl.update("opt", -1.0, 1)
        assert abs(fl.memory["opt"] - 0.0) < 0.01

    def test_accuracy(self):
        fl = FeedbackLoop(margin=0.0)
        fl.memory["x"] = 1.0
        pred = fl.predict("x")  # step 1, not forced explore
        fl.update("x", 1.0, pred)
        assert fl.accuracy() > 0.0

    def test_forced_explore(self):
        fl = FeedbackLoop(forced_explore_interval=3)
        fl.memory["x"] = 1.0
        fl.predict("x")  # step 1
        fl.predict("x")  # step 2
        pred = fl.predict("x")  # step 3 → forced explore
        assert pred == 0


class TestBatchAnalyzer:
    def test_analyze(self):
        ba = BatchAnalyzer()
        stats = ba.analyze([-1]*5 + [0]*3 + [1]*2)
        assert abs(stats["avoid"] - 0.5) < 0.01
        assert abs(stats["entropy"] - 1.4855) < 0.01  # log2(3) when uniform-ish

    def test_avoid_choose_ratio(self):
        ba = BatchAnalyzer()
        ba.add_batch([-1]*294 + [1])
        assert abs(ba.avoid_choose_ratio() - 294.0) < 1.0

    def test_conservation_std(self):
        ba = BatchAnalyzer()
        for _ in range(10):
            ba.add_batch([-1]*5 + [0]*3 + [1]*2)
        assert ba.conservation_std() < 0.01

    def test_population_summary(self):
        ba = BatchAnalyzer()
        for _ in range(5):
            ba.add_batch([-1]*5 + [0]*3 + [1]*2)
        summary = ba.population_summary()
        assert abs(summary["avoid"] - 0.5) < 0.01

    def test_empty(self):
        ba = BatchAnalyzer()
        assert ba.population_summary() == {}
        assert ba.avoid_choose_ratio() == 0.0
