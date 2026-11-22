"""
Pins down exactly how scripts/evaluate.py computes precision / recall / F1,
so the numbers it prints can be checked by hand.
"""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "evaluate", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


def _pred(frame, track_id, anomaly_type="running"):
    return {"frame": frame, "track_id": track_id, "anomaly_type": anomaly_type, "confidence": 1.0}


def test_hand_worked_example():
    # Ground truth: 4 anomalous (frame, track) pairs.
    gt = {
        (1, 1): "running",
        (2, 1): "running",
        (3, 1): "running",
        (5, 2): "loitering",
    }
    # Predictions: 3 hit GT pairs, 1 is a false alarm.
    # (5, 2) is a TP even though the type is wrong: matching is on (frame, track) only.
    preds = [
        _pred(1, 1),
        _pred(2, 1),
        _pred(5, 2, "running"),
        _pred(9, 3),
    ]
    b = evaluate.compute_metrics(preds, gt)["binary"]

    assert (b["tp"], b["fp"], b["fn"]) == (3, 1, 1)
    assert b["precision"] == pytest.approx(3 / 4)
    assert b["recall"] == pytest.approx(3 / 4)
    assert b["f1"] == pytest.approx(0.75)


def test_self_eval_mode_is_always_perfect():
    """Without --labels, evaluate.py uses its own predictions as ground truth."""
    preds = [_pred(f, 1) for f in range(10)] + [_pred(4, 2, "loitering")]
    gt = {(p["frame"], p["track_id"]): p["anomaly_type"] for p in preds}
    b = evaluate.compute_metrics(preds, gt)["binary"]

    assert b["fp"] == 0 and b["fn"] == 0
    assert b["f1"] == pytest.approx(1.0)


def test_no_predictions_gives_zero():
    b = evaluate.compute_metrics([], {(1, 1): "running"})["binary"]
    assert (b["tp"], b["fp"], b["fn"]) == (0, 0, 1)
    assert b["f1"] == pytest.approx(0.0)
