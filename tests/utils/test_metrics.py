from __future__ import annotations

import pytest

from miniprophet.utils.metrics import evaluate_submission, validate_ground_truth


def test_validate_ground_truth_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="Missing outcomes"):
        validate_ground_truth(["Yes", "No"], {"Yes": 1})


def test_evaluate_submission_returns_brier_score() -> None:
    result = evaluate_submission({"Yes": 0.8, "No": 0.2}, {"Yes": 1, "No": 0})
    assert "brier_score" in result
    assert result["brier_score"] == pytest.approx(0.04)
