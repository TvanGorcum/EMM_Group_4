"""Exceptional Model Mining (EMM) regression framework."""

from .pipeline import EMMRegressionPipeline, PipelineResult, ModelMetrics
from .schema import AttributeSpec, SubgroupSchema

__all__ = [
    "EMMRegressionPipeline",
    "PipelineResult",
    "ModelMetrics",
    "AttributeSpec",
    "SubgroupSchema",
]
