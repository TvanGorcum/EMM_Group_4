"""End-to-end orchestration of the EMM-enhanced regression workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from .emm import BeamSearchEMM, SubgroupResult
from .models import compute_metrics, fit_linear_regression
from .schema import SubgroupDescriptor, SubgroupSchema


@dataclass(frozen=True)
class ModelMetrics:
    """RMSE and R^2 metrics for a particular split."""

    rmse: float
    r2: float


@dataclass(frozen=True)
class PipelineResult:
    """Summary of the pipeline run containing metrics and mined subgroups."""

    baseline_train: ModelMetrics
    baseline_test: ModelMetrics
    augmented_train: ModelMetrics
    augmented_test: ModelMetrics
    subgroups: List[SubgroupResult]


class EMMRegressionPipeline:
    """Run EMM as preprocessing and augment linear regression with subgroup indicators."""

    def __init__(self, schema: SubgroupSchema) -> None:
        self._schema = schema
        self._subgroups: List[SubgroupResult] = []

    @property
    def subgroups(self) -> Sequence[SubgroupResult]:
        return self._subgroups

    def run(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        target: str,
        features: Sequence[str],
    ) -> PipelineResult:
        """Execute the full pipeline and return metrics."""
        baseline_train_artifacts = fit_linear_regression(train_frame[list(features)], train_frame[target])
        baseline_test_predictions = baseline_train_artifacts.model.predict(test_frame[list(features)])
        baseline_test_metrics = compute_metrics(
            test_frame[target], pd.Series(baseline_test_predictions, index=test_frame.index)
        )

        # Mine subgroups on training data using residuals from the baseline model
        residuals = train_frame[target] - baseline_train_artifacts.predictions
        emm = BeamSearchEMM(train_frame, residuals, self._schema)
        self._subgroups = emm.run()

        augmented_train, augmented_test = self._augment_frames(train_frame, test_frame)
        augmented_features = list(features) + [col for col in augmented_train.columns if col.startswith("sg_")]

        augmented_train_artifacts = fit_linear_regression(augmented_train[augmented_features], augmented_train[target])
        augmented_test_predictions = augmented_train_artifacts.model.predict(augmented_test[augmented_features])
        augmented_test_metrics = compute_metrics(
            augmented_test[target], pd.Series(augmented_test_predictions, index=augmented_test.index)
        )

        return PipelineResult(
            baseline_train=_as_model_metrics(baseline_train_artifacts.metrics),
            baseline_test=_as_model_metrics(baseline_test_metrics),
            augmented_train=_as_model_metrics(augmented_train_artifacts.metrics),
            augmented_test=_as_model_metrics(augmented_test_metrics),
            subgroups=self._subgroups,
        )

    def _augment_frames(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Append subgroup indicator columns to train and test frames."""
        train_augmented = train_frame.copy()
        test_augmented = test_frame.copy()

        for index, subgroup in enumerate(self._subgroups):
            descriptor: SubgroupDescriptor = subgroup.descriptor
            feature_name = descriptor.feature_token(index)
            train_augmented[feature_name] = subgroup.to_indicator(train_frame).astype(int)
            test_augmented[feature_name] = subgroup.to_indicator(test_frame).astype(int)

        return train_augmented, test_augmented


def _as_model_metrics(raw: Dict[str, float]) -> ModelMetrics:
    return ModelMetrics(rmse=raw["rmse"], r2=raw["r2"])
