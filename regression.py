#Train baselines(simple and more complex)
#'Extract' the subgroup specific models of subgroup_finder.py
#Make the models with subgroups as dummies

import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Any, Tuple
import numpy as np

from subgroup_finder import emm_beam_search

def train_basic_linear_regression(df, feature_cols = ['ECTS', 'GPA'], target_col = 'CalculatedNumericResult'):
    X = df[feature_cols]
    y = df[target_col]
    model = LinearRegression()
    model.fit(X, y)
    # Print basic model coefficients
    print("Coefficients (basic linear regression):")
    for col, coef in zip(feature_cols, model.coef_):
        print(f"  {col}: {coef}")
    print("Intercept (basic linear regression):", model.intercept_)
    return model

def train_complex_linear_regression(df, feature_cols = ['ECTS', 'GPA', 'course_repeater'], target_col = 'CalculatedNumericResult'): #This model still needs a lot of experimentation
    X = df[feature_cols]
    y = df[target_col]
    model = LinearRegression()
    model.fit(X, y)
    # Print basic model coefficients
    print("Coefficients (complex linear regression):")
    for col, coef in zip(feature_cols, model.coef_):
        print(f"  {col}: {coef}")
    print("Intercept (complex linear regression):", model.intercept_)
    return model

def collect_subgroup_models(
    df: pd.DataFrame,
    X_cols,
    y_col,
    attr_config,
    *,
    beam_width: int = 10,
    max_depth: int = 3,
    min_support: int = 100,
    top_S: int = 10,
) -> List[Dict[str, Any]]:
    results = emm_beam_search(
        df,
        X_cols=X_cols,
        y_col=y_col,
        attr_config=attr_config,
        beam_width=beam_width,
        max_depth=max_depth,
        min_support=min_support,
        top_S=top_S,
    )

    models = []
    for desc, D, mask, tbl_group, tbl_global in results:
        models.append({
            "description": desc,
            "n": int(mask.sum()),
            "cookD": float(D),
            # subgroup stats as dicts keyed by term name ("Intercept", feature names)
            "group_coef": tbl_group["coef"].to_dict(),
            "group_se":   tbl_group["se"].to_dict(),
            "group_t":    tbl_group["t"].to_dict(),
            "group_p":    tbl_group["p"].to_dict(),
            "group_sig":  tbl_group["sig"].to_dict(),
            # global stats
            "global_coef": tbl_global["coef"].to_dict(),
            "global_se":   tbl_global["se"].to_dict(),
            "global_t":    tbl_global["t"].to_dict(),
            "global_p":    tbl_global["p"].to_dict(),
            "global_sig":  tbl_global["sig"].to_dict(),
        })
    return models

#Converts the list from collect_subgroup_models() into a tidy long DataFrame with one row per (subgroup, term).
def models_to_long_dataframe(models: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for m in models:
        desc = m["description"]
        n = m["n"]
        cookD = m["cookD"]

        # terms from subgroup table (same index as global)
        for term, coef in m["group_coef"].items():
            records.append({
                "subgroup": desc,
                "n": n,
                "cookD": cookD,
                "term": term,
                "coef_group": coef,
                "se_group": m["group_se"][term],
                "t_group": m["group_t"][term],
                "p_group": m["group_p"][term],
                "sig_group": m["group_sig"][term],
                "coef_global": m["global_coef"][term],
                "se_global": m["global_se"][term],
                "t_global": m["global_t"][term],
                "p_global": m["global_p"][term],
                "sig_global": m["global_sig"][term],
            })
    return pd.DataFrame.from_records(records)


def save_models_csv(models: List[Dict[str, Any]], path: str) -> None:
    df_long = models_to_long_dataframe(models)
    df_long.to_csv(path, index=False)

def rebuild_models(models):
    """
    Convert the 'group_coef' data from each entry in models into
    sklearn LinearRegression objects, ready for prediction.
    Returns a dict: {description: (regressor, feature_order)}
    """
    model_objects = {}

    for m in models:
        desc = m["description"]
        coef_dict = m["group_coef"]

        # Extract intercept and coefficients
        intercept = coef_dict.get("Intercept", 0.0)
        # Remove intercept to get only features
        features = [k for k in coef_dict.keys() if k != "Intercept"]
        coefs = np.array([coef_dict[f] for f in features]).reshape(1, -1)

        # Build sklearn LinearRegression model
        reg = LinearRegression()
        reg.coef_ = coefs
        reg.intercept_ = intercept
        reg.feature_names_in_ = np.array(features)
        reg.n_features_in_ = len(features)

        model_objects[desc] = (reg, features)

    return model_objects

def final_estimator_with_coefs(model):
    """
    If model is a Pipeline, return the last step that has coef_.
    Otherwise return the model itself.
    """
    est = model
    if hasattr(model, "named_steps"):
        for name, step in reversed(list(model.named_steps.items())):
            if hasattr(step, "coef_"):
                est = step
                break
    elif hasattr(model, "steps"):
        for name, step in reversed(list(model.steps)):
            if hasattr(step, "coef_"):
                est = step
                break
    return est

def extract_linear_coefs(model, feature_names):
    """
    Return dict with intercept + per-feature coefficients.
    Dynamic column names: 'intercept' and 'coef__<feature_name>'.
    Falls back to None if model doesn't have coef_/intercept_.
    """
    est = final_estimator_with_coefs(model)
    intercept = getattr(est, "intercept_", None)
    coefs = getattr(est, "coef_", None)

    out = {"intercept": float(intercept) if intercept is not None else None}
    if coefs is None:
        out.update({f"coef__{f}": None for f in feature_names})
        return out

    coefs = np.ravel(coefs)
    for f, c in zip(feature_names, coefs):
        out[f"coef__{f}"] = float(c)
    return out

