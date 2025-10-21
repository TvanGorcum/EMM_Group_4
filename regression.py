import statsmodels.api as sm
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

from subgroup_finder import emm_beam_search

def train_linear_regression(df, feature_cols, target_col='CalculatedNumericResult'):
    """
    Train a statsmodels OLS regression with intercept.
    Returns the fitted model.
    """
    X = df[feature_cols]
    X = sm.add_constant(X)  # adds intercept
    y = df[target_col]
    model = sm.OLS(y, X).fit()
    # Print model coefficients
    for col, coef in zip(X.columns, model.params):
        print(f"  {col}: {coef}")
    return model

def collect_subgroup_models(
    df: pd.DataFrame,
    X_cols,
    y_col,
    attr_config,
    *,
    beam_width: int = 30,
    max_depth: int = 4,
    min_support: int = 70,
    top_S: int = 50,
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

def extract_linear_coefs(model, feature_names):
    """
    Return dict with intercept + per-feature coefficients and p-values.
    Dynamic column names: 'intercept', 'coef__<feature_name>', 'pval__<feature_name>'.
    """
    out = {}
    # Always use statsmodels
    out["intercept"] = float(model.params.get("const", model.params[0]))
    for f in feature_names:
        out[f"coef__{f}"] = float(model.params.get(f, float("nan")))
        out[f"pval__{f}"] = float(model.pvalues.get(f, float("nan")))
    return out

def add_subgroup_terms(df, description, base_cols, gamma_name=None):
    from evaluation import _description_to_mask
    mask = _description_to_mask(df, description)
    out = df.copy()
    gamma = gamma_name or f"gamma[{description}]"
    out[gamma] = mask.astype(int)
    inter_cols = []
    for x in base_cols:
        cname = f"{gamma}*{x}"
        out[cname] = out[gamma] * out[x]
        inter_cols.append(cname)
    return out, gamma, inter_cols

def _augment_with_kept(df, kept, base_cols):
    out = df.copy()
    for desc, gamma_name, inter_cols, _ in kept:
        if gamma_name not in out.columns:
            from evaluation import _description_to_mask
            mask = _description_to_mask(out, desc)
            out[gamma_name] = mask.astype(int)
        for x in base_cols:
            cname = f"{gamma_name}*{x}"
            if cname not in out.columns:
                out[cname] = out[gamma_name] * out[x]
    return out


