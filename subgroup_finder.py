import numpy as np
import pandas as pd
from itertools import product, combinations

# --- 0) Your Cook-based quality measure from earlier ---
def cooks_distance_emm(X, y, mask, add_intercept=True, use_pinv=True):
    """
    Compute the Cook-based quality measure (D_I) for Exceptional Model Mining.

    Implements:
        D_I = ((β_G - β)^T (X^T X) (β_G - β)) / (p * s^2)

    where:
        - β is the OLS estimate on ALL rows,
        - β_G is the OLS estimate using ONLY the subgroup rows (mask == True),
        - s^2 is the residual MSE from the global fit,
        - p is the number of regression coefficients (including intercept).

    Parameters
    ----------
    X : array-like, shape (N, d)
        Predictor matrix (no intercept column needed if add_intercept=True)
    y : array-like, shape (N,)
        Target vector
    mask : boolean array of shape (N,)
        True for rows in the subgroup to KEEP (the complement is "deleted")
    add_intercept : bool, default=True
        If True, adds an intercept column to X.
    use_pinv : bool, default=True
        If True, uses pseudoinverse (robust for singular or ill-conditioned X^TX).

    Returns
    -------
    dict with:
        - D : Cook's distance (float)
        - beta_global : global model coefficients
        - beta_group : subgroup model coefficients
        - s2 : global residual variance estimate
        - p : number of model parameters (including intercept)
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    mask = np.asarray(mask).astype(bool)

    if X.shape[0] != y.shape[0] or mask.shape[0] != y.shape[0]:
        raise ValueError("X, y, and mask must have the same number of rows.")

    # Add intercept if requested
    if add_intercept:
        X_design = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X_design = X

    N, p = X_design.shape

    # --- Fit global OLS model ---
    XTX = X_design.T @ X_design
    XTy = X_design.T @ y

    solve = np.linalg.pinv if use_pinv else np.linalg.inv
    beta_global = solve(XTX) @ XTy

    y_hat_global = X_design @ beta_global
    resid_global = y - y_hat_global
    df_resid = N - p
    if df_resid <= 0:
        raise ValueError("Not enough degrees of freedom.")
    s2 = float(resid_global.T @ resid_global) / df_resid

    # --- Fit subgroup OLS model (keep only True mask) ---
    if mask.sum() <= p:
        raise ValueError(f"Subgroup too small to fit: needs > {p} rows, has {mask.sum()}.")

    XG = X_design[mask]
    yG = y[mask]
    XGTXG = XG.T @ XG
    XGTyG = XG.T @ yG
    beta_group = solve(XGTXG) @ XGTyG

    # --- Compute Cook’s Distance D_I ---
    delta = beta_group - beta_global
    D = float(delta.T @ XTX @ delta) / (p * s2)

    return {
        "D": D,
        "beta_global": beta_global,
        "beta_group": beta_group,
        "s2": s2,
        "p": p,
    }


# --- 1) Helpers to create atomic conditions (pattern language) ---
def bin_numeric_series(s, n_bins=12):
    # equal-frequency binning; labels are intervals
    cats, bins = pd.qcut(s, q=n_bins, duplicates="drop", retbins=True)
    # produce atomic predicates: s <= b_k and s > b_k (or intervals)
    # here we’ll return interval conditions as strings for readability
    levels = cats.cat.categories
    return [("num_interval", s.name, iv) for iv in levels]  # iv is a pd.Interval

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

# --- 2) Apply a conjunction of atoms to get a boolean mask ---

def mask_from_atoms(df, atoms_subset):
    if not atoms_subset:
        return np.ones(len(df), dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    for pred, _ in atoms_subset:
        mask &= df.apply(pred, axis=1)
    return mask

# --- 3) Beam search over conjunctions ---

def emm_beam_search(
    df, X_cols, y_col, attr_config,
    beam_width=10, max_depth=2, min_support=100,
    top_S=10
):
    """
    Returns a list of (description, D, mask) for the top_S subgroups found.
    """
    X = df[X_cols].to_numpy()
    y = df[y_col].to_numpy()

    # Precompute atoms (depth-1 candidates)
    atoms = atomic_conditions(df, attr_config)

    # Score a set of candidates (each is a tuple of atoms)
    def score_candidates(cands):
        results = []
        for atoms_subset in cands:
            mask = mask_from_atoms(df, atoms_subset)
            if mask.sum() < min_support:
                continue
            try:
                D = cooks_distance_emm(X, y, mask)["D"]
            except Exception:
                continue
            desc = " ∧ ".join(lbl for _, lbl in atoms_subset)
            results.append((desc, D, mask.copy(), atoms_subset))
        # sort by Cook score desc
        results.sort(key=lambda t: t[1], reverse=True)
        return results

    # depth-1
    level1_cands = [ (a,) for a in atoms ]
    level1_scored = score_candidates(level1_cands)
    top_overall = level1_scored[:top_S]  # running top list
    frontier = level1_scored[:beam_width]  # beam for next expansions

    # depths ≥2
    for depth in range(2, max_depth + 1):
        next_cands = set()
        # refine: add one new atom not already present
        for _, _, _, atom_tuple in frontier:
            used = set(atom_tuple)
            for a in atoms:
                if a in used:
                    continue
                new_tuple = tuple(sorted(atom_tuple + (a,), key=lambda z: z[1]))  # sort by label for dedup
                next_cands.add(new_tuple)
        next_cands = list(next_cands)

        if not next_cands:
            break

        level_scored = score_candidates(next_cands)
        # update global top
        top_overall = sorted(top_overall + level_scored, key=lambda t: t[1], reverse=True)[:top_S]
        # new beam
        frontier = level_scored[:beam_width]

    # return compact results
    return [(desc, D, mask) for (desc, D, mask, _) in top_overall]

df = pd.read_csv('../data_final.csv')
numeric_cols = ["total_course_activities", "active_minutes", 'nr_distinct_files_viewed', 'nr_practice_exams_viewed']
df = df.copy()
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=numeric_cols).reset_index(drop=True)

# Suppose df is your DataFrame; choose predictors X_cols and target y_col
attr_config = {
    # declare which columns are used to DEFINE subgroups (attributes)
    # e.g. "Variety": "categorical", "Income": "numeric", ..
    #'faculty': 'categorical',
    'sex': 'categorical',
    'croho': 'categorical',
    'course_repeater': 'categorical',
    'ECTS' :'numeric',
    'GPA': 'numeric',
    'origin' : 'categorical'
}
top = emm_beam_search(
    df,
    X_cols=["total_course_activities", "active_minutes", 'nr_distinct_files_viewed', 'nr_practice_exams_viewed'],   # regressors for the model
    y_col="CalculatedNumericResult",                 # target
    attr_config=attr_config,   # how to define candidate subgroups
    beam_width=10,
    max_depth=3,
    min_support=100,
    top_S=10
)

for desc, D, mask in top:
    print(f"{desc}  -> Cook quality D={D:.3f}  (n={mask.sum()})")