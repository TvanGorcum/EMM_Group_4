import numpy as np
import pandas as pd
from itertools import product, combinations

#Help functions to label coefficients and coefficient tables
def label_coefs(beta, X_cols, add_intercept=True):
    names = (["Intercept"] if add_intercept else []) + list(X_cols)
    return pd.Series(beta, index=names)

def label_coef_table(beta, se, t, p, X_cols, add_intercept=True):
    names = (["Intercept"] if add_intercept else []) + list(X_cols)
    return pd.DataFrame(
        {"coef": beta, "se": se, "t": t, "p": p},
        index=names
    )

# 
def cooks_distance_emm(X, y, mask, add_intercept=True, use_pinv=True):
    """
    Compute Cook's distance D_I and return stats (β, se, t, p) for global and subgroup models.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    mask = np.asarray(mask).astype(bool)

    if X.shape[0] != y.shape[0] or mask.shape[0] != y.shape[0]:
        raise ValueError("X, y, and mask must have the same number of rows.")

    # Design matrices
    if add_intercept:
        X_design = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X_design = X

    N, p = X_design.shape

    # Global OLS with stats
    g = _ols_with_stats(X_design, y, use_pinv=use_pinv)

    # Subgroup OLS with stats
    if mask.sum() <= p:
        raise ValueError(f"Subgroup too small to fit: needs > {p} rows, has {mask.sum()}.")
    XG = X_design[mask]
    yG = y[mask]
    s = _ols_with_stats(XG, yG, use_pinv=use_pinv)

    # Cook's distance D_I: ((β_G - β)^T (X^T X) (β_G - β)) / (p * s_global^2)
    delta = s["beta"] - g["beta"]
    D = float(delta.T @ g["XTX"] @ delta) / (p * g["s2"])

    return {
        "D": D,
        # Global stats
        "beta_global": g["beta"],
        "se_global": g["se"],
        "t_global": g["t"],
        "p_global": g["p"],
        "s2_global": g["s2"],
        "df_global": g["df_resid"],
        # Subgroup stats
        "beta_group": s["beta"],
        "se_group": s["se"],
        "t_group": s["t"],
        "p_group": s["p"],
        "s2_group": s["s2"],
        "df_group": s["df_resid"],
        # model size
        "p": p,
    }


def bin_numeric_series(s, n_bins=4):
    """
    Given a numeric series, return a list of bins with constant interval width.
    """
    # equal-frequency binning
    cats, bins = pd.qcut(s, q=n_bins, duplicates="drop", retbins=True)
    # produce interval labels as ("num_interval", colname, pd.Interval)
    levels = cats.cat.categories
    return [("num_interval", s.name, iv) for iv in levels]  # iv is a pd.Interval

def _two_sided_p_from_t(tvals, df):
    """
    Return two-sided p-values for t statistics.
    Uses scipy.stats.t if available; otherwise normal approximation.
    """
    try:
        from scipy.stats import t as student_t
        # survival function *2 for two-sided
        return 2.0 * student_t.sf(np.abs(tvals), df)
    except Exception:
        # Normal approximation as fallback
        from math import erf, sqrt
        # p = 2 * (1 - Phi(|t|)) ; Phi via erf
        Phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        return 2.0 * (1.0 - np.vectorize(Phi)(np.abs(tvals)))

def _ols_with_stats(X_design, y, use_pinv=True):
    """
    OLS with β, s^2, df, standard errors, t-stats, and p-values.
    X_design MUST already include intercept if you want it.
    """
    N, p = X_design.shape
    XTX = X_design.T @ X_design
    XTy = X_design.T @ y

    inv = np.linalg.pinv if use_pinv else np.linalg.inv
    XTX_inv = inv(XTX)
    beta = XTX_inv @ XTy

    y_hat = X_design @ beta
    resid = y - y_hat

    df_resid = N - p
    if df_resid <= 0:
        raise ValueError("Not enough degrees of freedom.")
    s2 = float(resid.T @ resid) / df_resid

    # Var(β) = s^2 * (X'X)^-1 ; se = sqrt(diag(Var))
    var_beta = s2 * XTX_inv
    se = np.sqrt(np.clip(np.diag(var_beta), 0.0, np.inf))

    # Guard against zero SE
    with np.errstate(divide='ignore', invalid='ignore'):
        tvals = np.where(se > 0, beta / se, np.nan)

    pvals = _two_sided_p_from_t(tvals, df_resid)
    return {
        "beta": beta,
        "s2": s2,
        "df_resid": df_resid,
        "se": se,
        "t": tvals,
        "p": pvals,
        "XTX": XTX,           
    }


def atomic_conditions(df, attr_config):
    """
    attr_config: dict {col: "categorical"|"numeric"}
    Returns a list of callables f(row)->bool and readable labels.
    """
    atoms = []
    for col, kind in attr_config.items():
        if kind == "categorical":
            for v in df[col].dropna().unique():
                atoms.append( (lambda r, c=col, val=v: r[c] == val, f'{col}=={v!r}') )
        elif kind == "numeric":
            for (_, colname, iv) in bin_numeric_series(df[col]):
                # turn interval into predicate
                left = -np.inf if iv.left == -np.inf else iv.left
                right = np.inf if iv.right == np.inf else iv.right
                closed = iv.closed  # 'right' or 'left'
                def pred(r, c=colname, L=left, R=right, cl=closed):
                    x = r[c]
                    ok_left  = (x >= L) if cl in ("both", "left") else (x > L)
                    ok_right = (x <= R) if cl in ("both", "right") else (x < R)
                    return ok_left and ok_right
                atoms.append( (pred, f'{colname} in {iv}') )
        else:
            raise ValueError(f"Unknown kind for {col}: {kind}")
    return atoms

def mask_from_atoms(df, atoms_subset):
    """
    Given a list of (predicate, label) tuples, return a boolean mask for df rows
    that satisfy all predicates.
    If atoms_subset is empty, return all True.
    """
    if not atoms_subset:
        return np.ones(len(df), dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    for pred, _ in atoms_subset:
        mask &= df.apply(pred, axis=1)
    return mask


def emm_beam_search(
    df, X_cols, y_col, attr_config,
    beam_width=10, max_depth=3, min_support=100,
    top_S=10
):
    """
    Beam search for subgroups with high Cook's distance D_I.
    Returns a list of (description, D, mask, tbl_group, tbl_global).
    """
    X = df[X_cols].to_numpy()
    y = df[y_col].to_numpy()
    atoms = atomic_conditions(df, attr_config)

    def score_candidates(cands):
        results = []
        for atoms_subset in cands:
            mask = mask_from_atoms(df, atoms_subset)
            if mask.sum() < min_support:
                continue
            try:
                res = cooks_distance_emm(X, y, mask)

                D = res["D"]
                names = ["Intercept"] + list(X_cols)

                # subgroup coef table
                tbl_group = pd.DataFrame({
                    "coef": res["beta_group"],
                    "se":   res["se_group"],
                    "t":    res["t_group"],
                    "p":    res["p_group"],
                }, index=names)

                # add R-style significance stars (***, **, *, ., ' ')
                pvals = tbl_group["p"].values
                sig = np.full(pvals.shape, " ", dtype=object)
                sig[pvals <= 0.1]   = "."
                sig[pvals <= 0.05]  = "*"
                sig[pvals <= 0.01]  = "**"
                sig[pvals <= 0.001] = "***"
                tbl_group["sig"] = sig
                tbl_group = tbl_group[["coef","se","t","p","sig"]]

                # global coef table
                tbl_global = pd.DataFrame({
                    "coef": res["beta_global"],
                    "se":   res["se_global"],
                    "t":    res["t_global"],
                    "p":    res["p_global"],
                }, index=names)

                pvals_g = tbl_global["p"].values
                sig_g = np.full(pvals_g.shape, " ", dtype=object)
                sig_g[pvals_g <= 0.1]   = "."
                sig_g[pvals_g <= 0.05]  = "*"
                sig_g[pvals_g <= 0.01]  = "**"
                sig_g[pvals_g <= 0.001] = "***"
                tbl_global["sig"] = sig_g
                tbl_global = tbl_global[["coef","se","t","p","sig"]]

            except Exception:
                continue
            # description string
            desc = " ∧ ".join(lbl for _, lbl in atoms_subset)
            results.append((desc, D, mask.copy(), atoms_subset, tbl_group, tbl_global))
        results.sort(key=lambda t: t[1], reverse=True)
        return results

    # Start beam search
    level1_cands = [ (a,) for a in atoms ]
    level1_scored = score_candidates(level1_cands)
    top_overall = level1_scored[:top_S]
    frontier = level1_scored[:beam_width]

    for depth in range(2, max_depth + 1):
        next_cands = set()
        for _, _, _, atom_tuple, _, _ in frontier:
            used = set(atom_tuple)
            for a in atoms:
                if a in used:
                    continue
                new_tuple = tuple(sorted(atom_tuple + (a,), key=lambda z: z[1]))
                next_cands.add(new_tuple)
        next_cands = list(next_cands)
        if not next_cands:
            break

        level_scored = score_candidates(next_cands)
        top_overall = sorted(top_overall + level_scored, key=lambda t: t[1], reverse=True)[:top_S]
        frontier = level_scored[:beam_width]

    return [(desc, D, mask, tbl_group, tbl_global) for (desc, D, mask, _, tbl_group, tbl_global) in top_overall]

