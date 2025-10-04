"""Beam search Exceptional Model Mining implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import Predicate, SubgroupDescriptor, SubgroupSchema

# Fixed beam-search configuration (non-configurable as per framework design)
_BEAM_WIDTH = 10
_MAX_DEPTH = 3
_MIN_COVERAGE = 0.1
_TOP_K = 10


@dataclass(frozen=True)
class SubgroupResult:
    """Container for a mined subgroup and its quality statistics."""

    descriptor: SubgroupDescriptor
    coverage: int
    coverage_fraction: float
    mean_residual: float
    quality: float

    def to_indicator(self, frame: pd.DataFrame) -> pd.Series:
        """Return a boolean indicator of subgroup membership for each row."""
        return self.descriptor.apply(frame)

    def describe(self) -> Dict[str, object]:
        """Dictionary summary for reporting or serialization."""
        return {
            "description": self.descriptor.describe(),
            "coverage": self.coverage,
            "coverage_fraction": self.coverage_fraction,
            "mean_residual": self.mean_residual,
            "quality": self.quality,
        }


@dataclass
class _BeamNode:
    descriptor: SubgroupDescriptor
    mask: np.ndarray
    used_attributes: Tuple[str, ...]
    quality: float


class BeamSearchEMM:
    """Mine subgroups that explain systematic residuals via beam search."""

    def __init__(
        self,
        frame: pd.DataFrame,
        residuals: pd.Series,
        schema: SubgroupSchema,
    ) -> None:
        if len(frame) != len(residuals):
            raise ValueError("Frame and residuals must align.")
        self._frame = frame.reset_index(drop=True)
        self._residuals = residuals.reset_index(drop=True)
        self._schema = schema
        self._predicate_catalog = self._build_predicate_catalog(schema)

    @staticmethod
    def _build_predicate_catalog(schema: SubgroupSchema) -> Dict[str, List[Predicate]]:
        catalog: Dict[str, List[Predicate]] = {}
        for attribute in schema.attributes:
            catalog[attribute.name] = attribute.iter_predicates()
        return catalog

    def run(self) -> List[SubgroupResult]:
        n_rows = len(self._frame)
        residuals = self._residuals.to_numpy()
        all_results: List[SubgroupResult] = []

        # Start with the empty subgroup (covers everyone)
        initial_mask = np.ones(n_rows, dtype=bool)
        initial_descriptor = SubgroupDescriptor(())
        initial_node = _BeamNode(
            descriptor=initial_descriptor,
            mask=initial_mask,
            used_attributes=tuple(),
            quality=0.0,
        )

        current_beam: List[_BeamNode] = [initial_node]

        for depth in range(_MAX_DEPTH):
            next_beam: List[_BeamNode] = []
            candidates: List[_BeamNode] = []

            for node in current_beam:
                available_attributes = [
                    name for name in self._predicate_catalog.keys() if name not in node.used_attributes
                ]
                for attribute in available_attributes:
                    for predicate in self._predicate_catalog[attribute]:
                        new_mask = node.mask & predicate.apply(self._frame).to_numpy()
                        coverage = new_mask.sum()
                        coverage_fraction = coverage / n_rows if n_rows else 0.0
                        if coverage == 0 or coverage_fraction < _MIN_COVERAGE:
                            continue
                        subgroup_residuals = residuals[new_mask]
                        mean_residual = float(np.mean(subgroup_residuals))
                        quality = float(abs(mean_residual) * np.sqrt(coverage))
                        descriptor = SubgroupDescriptor(node.descriptor.predicates + (predicate,))
                        new_node = _BeamNode(
                            descriptor=descriptor,
                            mask=new_mask,
                            used_attributes=node.used_attributes + (attribute,),
                            quality=quality,
                        )
                        candidates.append(new_node)

                        all_results.append(
                            SubgroupResult(
                                descriptor=descriptor,
                                coverage=coverage,
                                coverage_fraction=coverage_fraction,
                                mean_residual=mean_residual,
                                quality=quality,
                            )
                        )

            # Select the next beam based on quality
            if not candidates:
                break
            candidates.sort(key=lambda node: node.quality, reverse=True)
            next_beam = candidates[:_BEAM_WIDTH]
            current_beam = next_beam

        # Keep only the global top-k subgroups
        all_results.sort(key=lambda result: result.quality, reverse=True)
        return all_results[:_TOP_K]
