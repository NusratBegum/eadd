import numpy as np
import pytest

from eadd import EADD
from eadd.synthetic import drift_stream


def run(pattern, **detector_kwargs):
    stream, truth = drift_stream(pattern, n_samples=3000, drift_at=1500, width=400, seed=1)
    detector = EADD(n_permutations=50, seed=7, **detector_kwargs)
    detections = [t for t, x in enumerate(stream) if detector.update(x)]
    return detector, detections, truth


def test_no_alarm_on_stable_stream():
    _, detections, _ = run("stable")
    assert detections == []


def test_detects_abrupt_drift_after_it_starts():
    detector, detections, truth = run("abrupt")
    assert detections, "abrupt mean shift was not detected"
    assert detections[0] >= truth[0]
    assert detections[0] - truth[0] <= 400


def test_report_after_detection_names_drifting_features():
    stream, _ = drift_stream("abrupt", n_samples=3000, drift_at=1500, seed=1)
    detector = EADD(n_permutations=50, seed=7)
    assert any(detector.update(x) for x in stream)  # stops at the first detection
    report = detector.get_last_report()
    assert report["auc"] >= 0.7
    assert report["p_value"] < 0.05
    top_two = list(report["feature_importances"])[:2]
    assert set(top_two) == {"f0", "f1"}  # only f0 and f1 drift


def test_reference_window_fills_before_testing():
    detector = EADD(n_reference_samples=100, n_current_samples=50, seed=0)
    for i in range(100):
        assert detector.update({"a": float(i), "b": 0.0}) is False
    assert len(detector.reference_window) == 100
    assert detector.current_window == []


def test_same_seed_is_reproducible():
    _, first, _ = run("incremental")
    _, second, _ = run("incremental")
    assert first == second


def test_unknown_pattern_rejected():
    with pytest.raises(ValueError):
        drift_stream("sideways")
