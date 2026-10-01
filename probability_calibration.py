"""Empirical probability calibration from completed out-of-sample observations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ProbabilityObservation:
    predicted_probability: Decimal
    outcome_win: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.predicted_probability, Decimal)
            or not self.predicted_probability.is_finite()
            or not Decimal("0") <= self.predicted_probability <= Decimal("1")
        ):
            raise ValueError("Observed probability must be a finite decimal between zero and one")
        if not isinstance(self.outcome_win, bool):
            raise ValueError("Outcome must be a boolean win/loss label")


class ProbabilityCalibrator:
    def __init__(self, *, bucket_width: Decimal = Decimal("0.1"), minimum_samples: int = 20):
        if (
            not isinstance(bucket_width, Decimal)
            or not bucket_width.is_finite()
            or bucket_width <= 0
            or bucket_width > 1
            or Decimal("1") % bucket_width != 0
        ):
            raise ValueError("Bucket width must be a positive decimal divisor of one")
        if minimum_samples < 1:
            raise ValueError("Minimum calibration sample count must be positive")
        self.bucket_width = bucket_width
        self.minimum_samples = minimum_samples

    def calibrate(
        self,
        predicted_probability: Decimal,
        observations: list[ProbabilityObservation],
    ) -> dict:
        if (
            not isinstance(predicted_probability, Decimal)
            or not predicted_probability.is_finite()
            or not Decimal("0") <= predicted_probability <= Decimal("1")
        ):
            raise ValueError("Predicted probability must be a finite decimal between zero and one")

        bucket_start = min(
            (predicted_probability // self.bucket_width) * self.bucket_width,
            Decimal("1") - self.bucket_width,
        )
        bucket_end = bucket_start + self.bucket_width
        matching = [
            observation for observation in observations
            if bucket_start <= observation.predicted_probability < bucket_end
            or (bucket_end == Decimal("1") and observation.predicted_probability == Decimal("1"))
        ]
        sample_count = len(matching)
        wins = sum(observation.outcome_win for observation in matching)
        calibrated_probability = Decimal(wins) / Decimal(sample_count) if sample_count >= self.minimum_samples else None
        brier_score = (
            sum(
                (observation.predicted_probability - Decimal(int(observation.outcome_win))) ** 2
                for observation in matching
            ) / Decimal(sample_count)
            if sample_count
            else None
        )
        return {
            "predicted_probability": predicted_probability,
            "calibrated_probability": calibrated_probability,
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "sample_count": sample_count,
            "minimum_samples": self.minimum_samples,
            "brier_score": brier_score,
            "status": "CALIBRATED" if calibrated_probability is not None else "INSUFFICIENT_SAMPLES",
        }