"""Linear regression utilities for the EMM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class RegressionArtifacts:
    """Store trained model and derived metrics."""

    model: LinearRegression
    predictions: pd.Series
    metrics: Dict[str, float]


def fit_linear_regression(features: pd.DataFrame, target: pd.Series) -> RegressionArtifacts:
    """Fit an ordinary least squares regression model."""
    regressor = LinearRegression()
    regressor.fit(features, target)
    predictions = pd.Series(regressor.predict(features), index=target.index)
    metrics = compute_metrics(target, predictions)
    return RegressionArtifacts(model=regressor, predictions=predictions, metrics=metrics)


def compute_metrics(target: pd.Series, predictions: pd.Series) -> Dict[str, float]:
    """Return RMSE and R^2 metrics."""
    mse = mean_squared_error(target, predictions)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(target, predictions))
    return {"rmse": rmse, "r2": r2}
