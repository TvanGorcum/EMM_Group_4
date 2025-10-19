import statsmodels.api as sm
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

from subgroup_finder import emm_beam_search

def train_basic_linear_regression(df, feature_cols=['ECTS', 'GPA'], target_col='CalculatedNumericResult'):
    X = df[feature_cols]
    X = sm.add_constant(X)  # adds intercept
    y = df[target_col]
    model = sm.OLS(y, X).fit()
    # Print basic model coefficients
    for col, coef in zip(['Intercept'] + feature_cols, model.params):
        print(f"  {col}: {coef}")
    return model

def train_complex_linear_regression(df, feature_cols=['ECTS', 'GPA', 'course_repeater'], target_col='CalculatedNumericResult'):
    X = df[feature_cols]
    X = sm.add_constant(X)  # adds intercept
    y = df[target_col]
    model = sm.OLS(y, X).fit()
    # Print complex model coefficients
    for col, coef in zip(['Intercept'] + feature_cols, model.params):
        print(f"  {col}: {coef}")
    return model

def collect_subgroup_models(
    df: pd.DataFrame,
    X_cols,
    y_col,
    attr_config,
    *,
    beam_width: int = 15,
    max_depth: int = 3,
    min_support: int = 150,
    top_S: int = 15,
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

def rebuild_models(models):
    """
    Convert the 'group_coef' data from each entry in models into
    statsmodels OLS objects, ready for prediction.
    Returns a dict: {description: (regressor, feature_order)}
    """
    model_objects = {}

    for m in models:
        desc = m["description"]
        coef_dict = m["group_coef"]

        # Extract intercept and coefficients
        intercept = coef_dict.get("Intercept", 0.0)
        features = [k for k in coef_dict.keys() if k != "Intercept"]
        # Create a dummy statsmodels-like object for compatibility
        # (for real prediction, you should refit or use params directly)
        reg = {"intercept": intercept, "coefs": {f: coef_dict[f] for f in features}}
        model_objects[desc] = (reg, features)

    return model_objects

def final_estimator_with_coefs(model):
    """
    For statsmodels, just return the model itself.
    """
    return model

def extract_linear_coefs(model, feature_names):
    """
    Return dict with intercept + per-feature coefficients and p-values.
    Dynamic column names: 'intercept', 'coef__<feature_name>', 'pval__<feature_name>'.
    """
    est = final_estimator_with_coefs(model)
    out = {}
    # For statsmodels
    if hasattr(est, "params") and hasattr(est, "pvalues"):
        out["intercept"] = float(est.params.get("const", est.params[0])) if "const" in est.params.index or est.params.index[0] == "const" else float(est.params[0])
        for f in feature_names:
            out[f"coef__{f}"] = float(est.params.get(f, float("nan")))
            out[f"pval__{f}"] = float(est.pvalues.get(f, float("nan")))
    else:
        # fallback for dict-like regressor
        out["intercept"] = model.get("intercept", None)
        for f in feature_names:
            out[f"coef__{f}"] = model["coefs"].get(f, None)
            out[f"pval__{f}"] = None
    return out

def _ensure_2d(a):
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else a

def _design_matrix(df, cols, add_intercept=True):
    X = df[cols].to_numpy()
    names = cols[:]
    if add_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])
        names = ["Intercept"] + names
    return X, names

def _ols_with_stats_matrix(X, y):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, p = X.shape
    XTX_inv = np.linalg.pinv(X.T @ X)
    beta = XTX_inv @ (X.T @ y)
    resid = y - X @ beta
    df_resid = n - p
    s2 = float(resid.T @ resid) / df_resid
    var_beta = s2 * XTX_inv
    se = np.sqrt(np.clip(np.diag(var_beta), 0.0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0, beta / se, np.nan)
    try:
        from scipy.stats import t as student_t
        pvals = 2.0 * student_t.sf(np.abs(tvals), df_resid)
    except Exception:
        from math import erf, sqrt
        Phi = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
        pvals = 2.0 * (1.0 - np.vectorize(Phi)(np.abs(tvals)))
    rss = float(resid.T @ resid)
    return {
        "beta": beta,
        "se": se,
        "t": tvals,
        "p": pvals,
        "rss": rss,
        "df_resid": df_resid,
    }

def partial_f_test(y, X_reduced, X_full):
    fit_r = _ols_with_stats_matrix(X_reduced, y)
    fit_f = _ols_with_stats_matrix(X_full, y)
    rss_r, rss_f = fit_r["rss"], fit_f["rss"]
    df_f = fit_f["df_resid"]
    q = X_full.shape[1] - X_reduced.shape[1]
    F = ((rss_r - rss_f) / q) / (rss_f / df_f)
    try:
        from scipy.stats import f as fdist
        p = fdist.sf(F, q, df_f)
    except Exception:
        p = np.nan
    return float(F), float(p), int(q), int(df_f)

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
