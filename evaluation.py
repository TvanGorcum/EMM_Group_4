#Define functions that will evualuate:
#1. The models seperate for each subgroup
#2. The baseline models and the global model with subgroups as a dummy only
from typing import List, Dict
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#We still need to settle and explain appropriate
def evaluate_linear_model(
    model,
    df: pd.DataFrame,
    X_cols: List[str],
    y_col: str
) -> Dict[str, float]:
    X = df[X_cols].values
    y = df[y_col].values
    y_pred = model.predict(X)

    return {
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "mse": mean_squared_error(y, y_pred)
    }
