"""Schema primitives for describing candidate EMM predicates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PredicateValue = Union[str, int, float]
CategoryDefinition = Union[PredicateValue, Sequence[PredicateValue]]
NumericBin = Tuple[Optional[float], Optional[float]]


@dataclass(frozen=True)
class Predicate:
    """Base class for attribute predicates."""

    attribute: str

    def apply(self, frame: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def token(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class CategoricalPredicate(Predicate):
    """Predicate that matches rows where the attribute takes one of the categories."""

    categories: Tuple[PredicateValue, ...]

    def apply(self, frame: pd.DataFrame) -> pd.Series:
        values = frame[self.attribute]
        return values.isin(self.categories)

    def describe(self) -> str:
        formatted = ", ".join(map(str, self.categories))
        return f"{self.attribute} in {{{formatted}}}"

    def token(self) -> str:
        safe = [str(cat).replace(" ", "_") for cat in self.categories]
        joined = "_or_".join(safe)
        return f"{self.attribute}_in_{joined}"


@dataclass(frozen=True)
class NumericRangePredicate(Predicate):
    """Predicate that matches rows within an interval."""

    lower: Optional[float]
    upper: Optional[float]
    include_lower: bool = True
    include_upper: bool = False

    def apply(self, frame: pd.DataFrame) -> pd.Series:
        values = frame[self.attribute]
        mask = pd.Series(np.ones(len(frame), dtype=bool), index=frame.index)
        if self.lower is not None:
            mask &= values >= self.lower if self.include_lower else values > self.lower
        if self.upper is not None:
            mask &= values <= self.upper if self.include_upper else values < self.upper
        return mask

    def describe(self) -> str:
        lower = "-inf" if self.lower is None else f"{self.lower:g}"
        upper = "inf" if self.upper is None else f"{self.upper:g}"
        left_bracket = "[" if self.include_lower else "("
        right_bracket = "]" if self.include_upper else ")"
        return f"{self.attribute} in {left_bracket}{lower}, {upper}{right_bracket}"

    def token(self) -> str:
        lower = "min" if self.lower is None else str(self.lower).replace(".", "p")
        upper = "max" if self.upper is None else str(self.upper).replace(".", "p")
        left = "ge" if self.include_lower else "gt"
        right = "le" if self.include_upper else "lt"
        return f"{self.attribute}_{left}_{lower}_{right}_{upper}"


@dataclass(frozen=True)
class AttributeSpec:
    """Specification of an attribute that can appear in subgroup descriptors."""

    name: str
    type: str  # "categorical" or "numeric"
    categories: Tuple[Tuple[PredicateValue, ...], ...] = field(default_factory=tuple)
    bins: Tuple[NumericBin, ...] = field(default_factory=tuple)

    @staticmethod
    def categorical(name: str, categories: Iterable[CategoryDefinition]) -> "AttributeSpec":
        normalised: List[Tuple[PredicateValue, ...]] = []
        for category in categories:
            if isinstance(category, Sequence) and not isinstance(category, (str, bytes)):
                normalised.append(tuple(category))
            else:
                normalised.append((category,))
        return AttributeSpec(name=name, type="categorical", categories=tuple(normalised))

    @staticmethod
    def numeric(name: str, bins: Iterable[NumericBin]) -> "AttributeSpec":
        return AttributeSpec(name=name, type="numeric", bins=tuple(bins))

    def iter_predicates(self) -> List[Predicate]:
        if self.type == "categorical":
            return [CategoricalPredicate(self.name, cats) for cats in self.categories]
        if self.type == "numeric":
            predicates: List[Predicate] = []
            for lower, upper in self.bins:
                predicates.append(
                    NumericRangePredicate(
                        attribute=self.name,
                        lower=lower,
                        upper=upper,
                        include_lower=True,
                        include_upper=upper is None,
                    )
                )
            return predicates
        raise ValueError(f"Unsupported attribute type: {self.type}")


@dataclass(frozen=True)
class SubgroupDescriptor:
    """Conjunction of predicates describing a subgroup."""

    predicates: Tuple[Predicate, ...]

    def apply(self, frame: pd.DataFrame) -> pd.Series:
        if not self.predicates:
            return pd.Series(np.ones(len(frame), dtype=bool), index=frame.index)
        mask = pd.Series(np.ones(len(frame), dtype=bool), index=frame.index)
        for predicate in self.predicates:
            mask &= predicate.apply(frame)
        return mask

    def describe(self) -> str:
        if not self.predicates:
            return "<entire population>"
        return " AND ".join(predicate.describe() for predicate in self.predicates)

    def feature_token(self, index: int) -> str:
        if not self.predicates:
            return f"sg_all_{index}"
        parts = [predicate.token() for predicate in self.predicates]
        token = "_and_".join(parts)
        return f"sg_{token}_{index}"


@dataclass(frozen=True)
class SubgroupSchema:
    """Collection of attribute specifications allowed in the descriptive space."""

    attributes: Tuple[AttributeSpec, ...]

    @staticmethod
    def from_attributes(attributes: Iterable[AttributeSpec]) -> "SubgroupSchema":
        return SubgroupSchema(attributes=tuple(attributes))

    def all_predicates(self) -> List[Predicate]:
        predicates: List[Predicate] = []
        for attribute in self.attributes:
            predicates.extend(attribute.iter_predicates())
        return predicates

    def attribute_names(self) -> List[str]:
        return [attribute.name for attribute in self.attributes]
