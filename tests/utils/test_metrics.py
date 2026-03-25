from __future__ import annotations

import pytest

from miniprophet.utils.metrics import (
    evaluate_submission,
    normalize_ground_truth,
    validate_ground_truth,
)


def test_validate_ground_truth_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="keys must be"):
        validate_ground_truth({"Yes": 1})


def test_evaluate_submission_returns_brier_score() -> None:
    result = evaluate_submission({"Yes": 0.8, "No": 0.2}, {"Yes": 1, "No": 0})
    assert "brier_score" in result
    assert result["brier_score"] == pytest.approx(0.04)


def test_normalize_ground_truth_bare_int() -> None:
    assert normalize_ground_truth(1) == {"Yes": 1, "No": 0}
    assert normalize_ground_truth(0) == {"Yes": 0, "No": 1}


def test_normalize_ground_truth_bare_float() -> None:
    assert normalize_ground_truth(1.0) == {"Yes": 1, "No": 0}
    assert normalize_ground_truth(0.0) == {"Yes": 0, "No": 1}


def test_normalize_ground_truth_partial_dict() -> None:
    assert normalize_ground_truth({"Yes": 1}) == {"Yes": 1, "No": 0}


def test_normalize_ground_truth_full_dict() -> None:
    assert normalize_ground_truth({"Yes": 1, "No": 0}) == {"Yes": 1, "No": 0}


def test_normalize_ground_truth_case_insensitive() -> None:
    assert normalize_ground_truth({"yes": 1}) == {"Yes": 1, "No": 0}
    assert normalize_ground_truth({"YES": 0, "NO": 1}) == {"Yes": 0, "No": 1}


def test_normalize_ground_truth_rejects_invalid_int() -> None:
    with pytest.raises(ValueError, match="must be 0 or 1"):
        normalize_ground_truth(2)


def test_normalize_ground_truth_rejects_bad_key() -> None:
    with pytest.raises(ValueError, match="must be 'Yes' or 'No'"):
        normalize_ground_truth({"Maybe": 1})


def test_normalize_ground_truth_rejects_non_complementary() -> None:
    with pytest.raises(ValueError, match="complementary"):
        normalize_ground_truth({"Yes": 1, "No": 1})


def test_normalize_ground_truth_rejects_bad_type() -> None:
    with pytest.raises(ValueError, match="must be int, float, or dict"):
        normalize_ground_truth("yes")  # type: ignore[arg-type]
