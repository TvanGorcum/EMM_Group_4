"""Different Slopes for Different Folks implementation for RTDM.csv.
"""
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy import linalg as la

# Global configuration parameters ------------------------------------------------
COURSE_CODE = "2IAB1"
TARGET = "EndResult"
MIN_SUPPORT = 100
MAX_DEPTH = 2
BEAM_WIDTH = 10
TOP_K = 20
N_BINS = 12

DO_CV = True
N_FOLDS = 5
TEST_SIZE = 0.2

# Metrics layer (edit here only)
METRIC_FUNCS = {
    "rmse": lambda y, yhat: float(np.sqrt(np.mean((y - yhat) ** 2))),
    "mae": lambda y, yhat: float(np.mean(np.abs(y - yhat))),
    "r2": lambda y, yhat: float(
        1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
        if np.sum((y - np.mean(y)) ** 2) > 0
        else np.nan
    ),
}
SELECTED_METRICS = ["rmse", "mae", "r2"]
BETTER_DIRECTION = {"rmse": "lower", "mae": "lower", "r2": "higher"}

np.random.seed(42)


@dataclass
class GlobalOLSResult:
    beta: np.ndarray
    se: np.ndarray
    r2: float
    adj_r2: float
    residuals: np.ndarray
    s2: float
    hat_diag: np.ndarray
    xtx_inv: np.ndarray
    xtx: np.ndarray
    design_matrix: np.ndarray
    feature_names: List[str]


@dataclass
class SubgroupModelSummary:
    beta: np.ndarray
    se: np.ndarray
    r2: Optional[float]
    adj_r2: Optional[float]


@dataclass
class CandidateResult:
    rule: Tuple[Tuple[str, str], ...]
    depth: int
    support: int
    cook_score: float
    score_type: str
    time_ms: float
    exact: bool
    summary: Optional[SubgroupModelSummary] = None


# Utility functions --------------------------------------------------------------
def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def prepare_data(csv_path: Path, course_code: str, target: str) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, str]]:
    """Load and clean the RTDM dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset at {csv_path}")

    df_raw = pd.read_csv(csv_path)
    print_header("Initial Data Snapshot")
    print(f"Rows x Columns (raw): {df_raw.shape}")

    df = df_raw.copy()
    df = df[df.get("course_code") == course_code]
    print(f"Rows x Columns after course filter: {df.shape}")

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    before_dropna = len(df)
    df = df.dropna(subset=[target])
    print(f"Dropped {before_dropna - len(df)} rows with NaN target")

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df.insert(0, "original_index", df.index)
    print(f"Rows x Columns after cleaning: {df.shape}")

    if len(df) < 100:
        raise AssertionError("Insufficient rows after filtering; need at least 100.")

    end_result_summary = df[target].describe()
    print("EndResult summary:")
    print(end_result_summary)

    drop_cols: List[str] = []
    for col in df.columns:
        lower = col.lower()
        if col == target:
            continue
        if lower.endswith("id") or lower.startswith("id") or "_id" in lower or lower.endswith("code"):
            drop_cols.append(col)
    drop_cols = sorted(set(drop_cols))
    if "course_code" in df.columns:
        drop_cols.append("course_code")
    if "Passed" in df.columns:
        drop_cols.append("Passed")
    drop_cols = sorted(set(drop_cols))

    modeling_df = df.drop(columns=drop_cols, errors="ignore")

    numeric_features = [
        col
        for col in modeling_df.columns
        if col != target and pd.api.types.is_numeric_dtype(modeling_df[col])
    ]
    categorical_features = [
        col
        for col in modeling_df.columns
        if col != target and (pd.api.types.is_object_dtype(modeling_df[col]) or pd.api.types.is_bool_dtype(modeling_df[col]))
    ]

    if "original_index" in numeric_features:
        numeric_features.remove("original_index")
    if "original_index" in categorical_features:
        categorical_features.remove("original_index")

    rare_threshold = 20
    for col in categorical_features:
        counts = modeling_df[col].value_counts(dropna=False)
        rare_values = counts[counts < rare_threshold].index
        if len(rare_values) > 0:
            modeling_df[col] = modeling_df[col].where(~modeling_df[col].isin(rare_values), other="other")

    print_header("Modeling Features")
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    binned_mapping = make_binned_attributes(modeling_df, numeric_features, n_bins=N_BINS)
    print_header("Binned Attributes")
    for orig, binned in binned_mapping.items():
        print(f"{orig} -> {binned}")

    feature_config = {
        "numeric": numeric_features,
        "categorical": categorical_features,
    }
    return modeling_df, feature_config, binned_mapping


def make_binned_attributes(df: pd.DataFrame, numeric_cols: Sequence[str], n_bins: int = 12) -> Dict[str, str]:
    """Create quantile-based bins for numeric attributes."""
    mapping: Dict[str, str] = {}
    for col in numeric_cols:
        binned_name = f"{col}_bin"
        series = df[col]
        if series.dropna().nunique() < 2:
            df[binned_name] = pd.NA
        else:
            try:
                df[binned_name] = pd.qcut(series, q=min(n_bins, series.dropna().nunique()), labels=False, duplicates="drop")
            except ValueError:
                df[binned_name] = pd.NA
        mapping[col] = binned_name
    return mapping


def build_preprocessor(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
        transformers.append(("num", num_pipeline, list(numeric_features)))
    if categorical_features:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "ohe",
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False),
                ),
            ]
        )
        transformers.append(("cat", cat_pipeline, list(categorical_features)))
    if not transformers:
        raise ValueError("No features available for modeling.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    feature_names: List[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe: OneHotEncoder = trans.named_steps["ohe"]
            feature_names.extend(ohe.get_feature_names_out(cols).tolist())
    return feature_names


def fit_ols_qr(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float, np.ndarray]:
    n_samples, n_features = X.shape
    try:
        q, r = np.linalg.qr(X, mode="reduced")
        beta = np.linalg.solve(r, q.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    df_resid = n_samples - n_features
    ss_res = residuals.T @ residuals
    if df_resid > 0:
        s2 = float(ss_res / df_resid)
    else:
        s2 = np.nan
    try:
        xtx = X.T @ X
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx = X.T @ X
        xtx_inv = np.linalg.pinv(xtx)
    if df_resid > 0:
        se = np.sqrt(np.diag(xtx_inv) * s2)
    else:
        se = np.full(n_features, np.nan)
    tss = np.sum((y - y.mean()) ** 2)
    if tss > 0:
        r2 = 1 - ss_res / tss
    else:
        r2 = np.nan
    if df_resid > 0 and n_samples > 1:
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / df_resid
    else:
        adj_r2 = np.nan
    return beta, residuals, r2, adj_r2, s2, xtx_inv


def build_global_ols(df: pd.DataFrame, feature_config: Dict[str, List[str]], target: str) -> Tuple[GlobalOLSResult, ColumnTransformer, pd.DataFrame]:
    preprocessor = build_preprocessor(feature_config["numeric"], feature_config["categorical"])
    X = preprocessor.fit_transform(df)
    feature_names = get_feature_names(preprocessor)
    y = df[target].to_numpy()

    intercept = np.ones((X.shape[0], 1))
    X_design = np.concatenate([intercept, X], axis=1)
    design_feature_names = ["intercept"] + feature_names

    beta, residuals, r2, adj_r2, s2, xtx_inv = fit_ols_qr(X_design, y)
    hat_diag = np.sum((X_design @ xtx_inv) * X_design, axis=1)
    se = np.sqrt(np.diag(xtx_inv) * s2) if not np.isnan(s2) else np.full(len(beta), np.nan)
    xtx = X_design.T @ X_design

    print_header("Global OLS Summary")
    print(f"R^2: {r2:.4f}, Adjusted R^2: {adj_r2:.4f}")

    result = GlobalOLSResult(
        beta=beta,
        se=se,
        r2=r2,
        adj_r2=adj_r2,
        residuals=residuals,
        s2=s2,
        hat_diag=hat_diag,
        xtx_inv=xtx_inv,
        xtx=xtx,
        design_matrix=X_design,
        feature_names=design_feature_names,
    )
    design_df = pd.DataFrame(X_design, columns=design_feature_names)
    design_df[target] = y
    return result, preprocessor, design_df


def extract_attributes(df: pd.DataFrame, binned_mapping: Dict[str, str], feature_config: Dict[str, List[str]]) -> pd.DataFrame:
    attr_cols: List[str] = []
    for col in feature_config["categorical"]:
        attr_cols.append(col)
    for col in feature_config["numeric"]:
        binned = binned_mapping.get(col)
        if binned and binned in df.columns:
            attr_cols.append(binned)
    df_attrs = df[attr_cols].copy()
    return df_attrs


def generate_candidates_level1(df_attrs: pd.DataFrame, min_support: int) -> List[Tuple[Tuple[str, str], ...]]:
    candidates: List[Tuple[Tuple[str, str], ...]] = []
    for col in df_attrs.columns:
        values = df_attrs[col].dropna().unique()
        for val in values:
            mask = df_attrs[col] == val
            support = int(mask.sum())
            if support >= min_support:
                candidates.append(((col, str(val)),))
    return candidates


def rule_mask(df_attrs: pd.DataFrame, rule: Tuple[Tuple[str, str], ...]) -> np.ndarray:
    mask = np.ones(len(df_attrs), dtype=bool)
    for col, val in rule:
        mask &= df_attrs[col].astype(str) == val
    return mask


def cook_bounds(indices_complement: np.ndarray, global_res: GlobalOLSResult) -> Tuple[Optional[float], str]:
    if indices_complement.size == 0:
        return None, "none"
    hat_vals = global_res.hat_diag[indices_complement]
    residuals = global_res.residuals[indices_complement]
    T = float(np.sum(hat_vals))
    if T >= 1:
        return None, "invalid"
    E = float(np.sum(residuals ** 2))
    p = len(global_res.beta)
    s2 = global_res.s2
    if s2 <= 0 or math.isnan(s2):
        return None, "invalid"
    bound = (T / ((1 - T) ** 2)) * (E / (p * s2))
    return bound, "bound_T"


def cook_score_exact(
    subgroup_indices: np.ndarray,
    df_design: pd.DataFrame,
    global_res: GlobalOLSResult,
) -> Tuple[float, SubgroupModelSummary]:
    X_sub = global_res.design_matrix[subgroup_indices]
    y_sub = df_design[TARGET].to_numpy()[subgroup_indices]
    beta_sub, residuals, r2_sub, adj_r2_sub, s2_sub, xtx_inv_sub = fit_ols_qr(X_sub, y_sub)

    diff = beta_sub - global_res.beta
    p = len(global_res.beta)
    cook = float((p / global_res.s2) * (diff.T @ global_res.xtx @ diff)) if global_res.s2 > 0 else np.nan
    se_sub = (
        np.sqrt(np.diag(xtx_inv_sub) * s2_sub)
        if s2_sub is not None and not np.isnan(s2_sub)
        else np.full(len(beta_sub), np.nan)
    )
    summary = SubgroupModelSummary(beta=beta_sub, se=se_sub, r2=r2_sub, adj_r2=adj_r2_sub)
    return cook, summary


def fit_eval_global_model(global_res: GlobalOLSResult, indices: np.ndarray) -> np.ndarray:
    return global_res.design_matrix[indices] @ global_res.beta


def specialize(rule: Tuple[Tuple[str, str], ...], new_atom: Tuple[str, str]) -> Tuple[Tuple[str, str], ...]:
    cols = {col for col, _ in rule}
    if new_atom[0] in cols:
        return rule
    new_rule = tuple(sorted(rule + (new_atom,), key=lambda x: x[0]))
    return new_rule


def beam_search(
    df_attrs: pd.DataFrame,
    global_res: GlobalOLSResult,
    df_design: pd.DataFrame,
    min_support: int,
    max_depth: int,
    beam_width: int,
) -> Tuple[List[CandidateResult], Dict[int, List[CandidateResult]]]:
    attr_values: Dict[str, List[str]] = {}
    for col in df_attrs.columns:
        values = sorted(df_attrs[col].dropna().astype(str).unique().tolist())
        attr_values[col] = values

    candidates_level1 = generate_candidates_level1(df_attrs, min_support)
    results_all: List[CandidateResult] = []
    depth_snapshots: Dict[int, List[CandidateResult]] = {}

    def evaluate_rule(rule: Tuple[Tuple[str, str], ...], depth: int, threshold: Optional[float]) -> CandidateResult:
        start = time.time()
        mask = rule_mask(df_attrs, rule)
        support = int(mask.sum())
        if support < min_support or support == 0 or support == len(df_attrs):
            elapsed = (time.time() - start) * 1000
            return CandidateResult(rule, depth, support, float("nan"), "insufficient", elapsed, False)
        complement_idx = np.where(~mask)[0]
        bound, bound_type = cook_bounds(complement_idx, global_res)
        if threshold is not None and bound is not None and not math.isnan(bound) and bound < threshold:
            elapsed = (time.time() - start) * 1000
            return CandidateResult(rule, depth, support, bound, f"bound:{bound_type}", elapsed, False)
        cook_score, summary = cook_score_exact(np.where(mask)[0], df_design, global_res)
        elapsed = (time.time() - start) * 1000
        candidate = CandidateResult(rule, depth, support, cook_score, "exact", elapsed, True)
        candidate.summary = summary  # type: ignore[attr-defined]
        return candidate

    exact_scores_level: Dict[int, List[float]] = {depth: [] for depth in range(1, max_depth + 1)}

    level = 1
    level_results: List[CandidateResult] = []
    threshold = None
    for rule in candidates_level1:
        threshold = None
        if exact_scores_level[level]:
            sorted_scores = sorted(exact_scores_level[level], reverse=True)
            if len(sorted_scores) >= beam_width:
                threshold = sorted_scores[beam_width - 1]
        candidate = evaluate_rule(rule, level, threshold)
        level_results.append(candidate)
        if candidate.exact:
            exact_scores_level[level].append(candidate.cook_score)
            results_all.append(candidate)
    level_results_sorted = sorted(
        [c for c in level_results if c.exact], key=lambda c: c.cook_score, reverse=True
    )[:beam_width]
    depth_snapshots[level] = level_results_sorted

    current_beam = level_results_sorted

    for depth in range(2, max_depth + 1):
        next_candidates: List[CandidateResult] = []
        new_exact_scores: List[float] = []
        for parent in current_beam:
            rule_cols = {col for col, _ in parent.rule}
            for attr, values in attr_values.items():
                if attr in rule_cols:
                    continue
                for val in values:
                    new_rule = specialize(parent.rule, (attr, val))
                    if new_rule == parent.rule:
                        continue
                    threshold = None
                    if new_exact_scores:
                        sorted_scores = sorted(new_exact_scores, reverse=True)
                        if len(sorted_scores) >= beam_width:
                            threshold = sorted_scores[beam_width - 1]
                    candidate = evaluate_rule(new_rule, depth, threshold)
                    if candidate.exact:
                        new_exact_scores.append(candidate.cook_score)
                        results_all.append(candidate)
                    next_candidates.append(candidate)
        current_beam = sorted(
            [c for c in next_candidates if c.exact], key=lambda c: c.cook_score, reverse=True
        )[:beam_width]
        depth_snapshots[depth] = current_beam

    top_candidates = sorted(results_all, key=lambda c: c.cook_score, reverse=True)[:TOP_K]
    return top_candidates, depth_snapshots


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, metric_names: Sequence[str]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for name in metric_names:
        func = METRIC_FUNCS[name]
        results[name] = func(y_true, y_pred)
    return results


def drop_constant_columns(X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    keep_indices = [0]  # keep intercept
    updated_names = [feature_names[0]]
    for idx in range(1, X_train.shape[1]):
        col = X_train[:, idx]
        if np.allclose(col, col[0]):
            continue
        keep_indices.append(idx)
        updated_names.append(feature_names[idx])
    return X_train[:, keep_indices], X_test[:, keep_indices], updated_names


def drop_collinear_columns(
    X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str], tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if X_train.shape[1] <= 1:
        return X_train, X_test, feature_names
    try:
        _, r, piv = la.qr(X_train, mode="economic", pivoting=True)
    except Exception:
        return X_train, X_test, feature_names
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        return X_train, X_test, feature_names
    threshold = tol * diag.max()
    rank = int(np.sum(diag > threshold))
    if rank <= 0:
        keep_indices = [0]
    else:
        keep_indices = sorted(piv[:rank].tolist())
        if 0 not in keep_indices:
            keep_indices = [0] + [idx for idx in keep_indices if idx != 0]
    X_train_reduced = X_train[:, keep_indices]
    X_test_reduced = X_test[:, keep_indices]
    reduced_names = [feature_names[idx] for idx in keep_indices]
    return X_train_reduced, X_test_reduced, reduced_names


def evaluate_subgroup_models(
    candidate: CandidateResult,
    df: pd.DataFrame,
    df_attrs: pd.DataFrame,
    feature_config: Dict[str, List[str]],
    global_res: GlobalOLSResult,
    design_df: pd.DataFrame,
) -> Dict[str, object]:
    mask = rule_mask(df_attrs, candidate.rule)
    subgroup_df = df.loc[mask].copy()
    y = subgroup_df[TARGET].to_numpy()

    unstable_flag = False
    try:
        preprocessor_full = build_preprocessor(feature_config["numeric"], feature_config["categorical"])
        X_full = preprocessor_full.fit_transform(subgroup_df)
        n_features_after_ohe = X_full.shape[1]
    except ValueError:
        n_features_after_ohe = 0
        unstable_flag = True

    encoded_param_count = n_features_after_ohe + 1
    if subgroup_df.shape[0] < encoded_param_count + 10:
        unstable_flag = True

    metrics_global: List[Dict[str, float]] = []
    metrics_subgroup: List[Dict[str, float]] = []

    if unstable_flag:
        eval_result = {
            "rank": None,
            "rule": candidate.rule,
            "support": int(mask.sum()),
            "unstable_flag": True,
        }
        for metric in SELECTED_METRICS:
            eval_result[f"cv_{metric}_global_on_G"] = np.nan
            eval_result[f"cv_{metric}_subgroup"] = np.nan
            eval_result[f"delta_{metric}"] = np.nan
            eval_result[f"cv_{metric}_subgroup_std"] = np.nan
        eval_result["n_features_after_ohe"] = n_features_after_ohe
        eval_result["model_summary"] = candidate.summary  # type: ignore[attr-defined]
        return eval_result

    indices = np.where(mask)[0]
    if DO_CV and subgroup_df.shape[0] >= max(2, N_FOLDS):
        n_splits = min(N_FOLDS, subgroup_df.shape[0])
        if n_splits < 2:
            unstable_flag = True
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_idx, test_idx in splitter.split(subgroup_df):
                subgroup_train = subgroup_df.iloc[train_idx]
                subgroup_test = subgroup_df.iloc[test_idx]

                # Global model predictions
                y_test = subgroup_test[TARGET].to_numpy()
                y_pred_global = fit_eval_global_model(global_res, indices[test_idx])
                metrics_global.append(evaluate(y_test, y_pred_global, SELECTED_METRICS))

                # Subgroup-specific model
                preprocessor = build_preprocessor(feature_config["numeric"], feature_config["categorical"])
                X_train = preprocessor.fit_transform(subgroup_train)

                feature_names = ["intercept"] + get_feature_names(preprocessor)
                intercept_train = np.ones((X_train.shape[0], 1))
                X_train_design = np.concatenate([intercept_train, X_train], axis=1)

                X_test = preprocessor.transform(subgroup_test)
                intercept_test = np.ones((X_test.shape[0], 1))
                X_test_design = np.concatenate([intercept_test, X_test], axis=1)

                X_train_design, X_test_design, feature_names = drop_constant_columns(
                    X_train_design, X_test_design, feature_names
                )
                X_train_design, X_test_design, feature_names = drop_collinear_columns(
                    X_train_design, X_test_design, feature_names
                )
                if X_train_design.shape[1] >= X_train_design.shape[0]:
                    unstable_flag = True
                    break

                beta_fold, _, _, _, _, _ = fit_ols_qr(X_train_design, subgroup_train[TARGET].to_numpy())
                y_pred_subgroup = X_test_design @ beta_fold
                metrics_subgroup.append(evaluate(y_test, y_pred_subgroup, SELECTED_METRICS))
    else:
        train_df, test_df = train_test_split(
            subgroup_df, test_size=TEST_SIZE, random_state=42, shuffle=True
        )
        if len(train_df) <= 1 or len(test_df) <= 1:
            unstable_flag = True
        else:
            # Global predictions
            y_test = test_df[TARGET].to_numpy()
            test_indices = test_df.index.to_numpy()
            y_pred_global = fit_eval_global_model(global_res, test_indices)
            metrics_global.append(evaluate(y_test, y_pred_global, SELECTED_METRICS))

            preprocessor = build_preprocessor(feature_config["numeric"], feature_config["categorical"])
            X_train = preprocessor.fit_transform(train_df)
            feature_names = ["intercept"] + get_feature_names(preprocessor)
            X_train_design = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)
            X_test = preprocessor.transform(test_df)
            X_test_design = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)
            X_train_design, X_test_design, feature_names = drop_constant_columns(
                X_train_design, X_test_design, feature_names
            )
            X_train_design, X_test_design, feature_names = drop_collinear_columns(
                X_train_design, X_test_design, feature_names
            )
            if X_train_design.shape[1] >= X_train_design.shape[0]:
                unstable_flag = True
            else:
                beta_fold, _, _, _, _, _ = fit_ols_qr(X_train_design, train_df[TARGET].to_numpy())
                y_pred_subgroup = X_test_design @ beta_fold
                metrics_subgroup.append(evaluate(y_test, y_pred_subgroup, SELECTED_METRICS))

    eval_result = {
        "rank": None,
        "rule": candidate.rule,
        "support": int(mask.sum()),
        "unstable_flag": unstable_flag,
    }

    if unstable_flag or not metrics_global or not metrics_subgroup:
        for metric in SELECTED_METRICS:
            eval_result[f"cv_{metric}_global_on_G"] = np.nan
            eval_result[f"cv_{metric}_subgroup"] = np.nan
            eval_result[f"delta_{metric}"] = np.nan
            eval_result[f"cv_{metric}_subgroup_std"] = np.nan
    else:
        global_means = {metric: float(np.mean([m[metric] for m in metrics_global])) for metric in SELECTED_METRICS}
        subgroup_means = {metric: float(np.mean([m[metric] for m in metrics_subgroup])) for metric in SELECTED_METRICS}
        subgroup_stds = {metric: float(np.std([m[metric] for m in metrics_subgroup], ddof=1)) if len(metrics_subgroup) > 1 else 0.0 for metric in SELECTED_METRICS}
        for metric in SELECTED_METRICS:
            eval_result[f"cv_{metric}_global_on_G"] = global_means[metric]
            eval_result[f"cv_{metric}_subgroup"] = subgroup_means[metric]
            eval_result[f"cv_{metric}_subgroup_std"] = subgroup_stds[metric]
            delta = subgroup_means[metric] - global_means[metric]
            eval_result[f"delta_{metric}"] = delta
    eval_result["n_features_after_ohe"] = n_features_after_ohe
    eval_result["model_summary"] = candidate.summary  # type: ignore[attr-defined]
    return eval_result


def format_rule(rule: Tuple[Tuple[str, str], ...]) -> str:
    parts = [f"{col} == {val}" for col, val in rule]
    return " AND ".join(parts)


def main() -> None:
    csv_path = Path("/mnt/data/RTDM.csv")
    df, feature_config, binned_mapping = prepare_data(csv_path, COURSE_CODE, TARGET)

    df_attrs = extract_attributes(df, binned_mapping, feature_config)

    global_res, preprocessor, design_df = build_global_ols(df, feature_config, TARGET)

    print_header("Starting Beam Search")
    top_candidates, depth_snapshots = beam_search(
        df_attrs=df_attrs,
        global_res=global_res,
        df_design=design_df,
        min_support=MIN_SUPPORT,
        max_depth=MAX_DEPTH,
        beam_width=BEAM_WIDTH,
    )

    for depth, snapshot in depth_snapshots.items():
        print_header(f"Depth {depth} Beam Snapshot")
        snapshot_rows = [
            {
                "rule": format_rule(c.rule),
                "support": c.support,
                "cook_score": c.cook_score,
                "score_type": c.score_type,
            }
            for c in snapshot
        ]
        print(pd.DataFrame(snapshot_rows))

    print(f"Evaluated {len(top_candidates)} top candidates")

    records_ranked = []
    for idx, cand in enumerate(top_candidates, start=1):
        records_ranked.append(
            {
                "rank": idx,
                "rule": format_rule(cand.rule),
                "depth": cand.depth,
                "support": cand.support,
                "cook_score": cand.cook_score,
                "score_type": cand.score_type,
                "time_ms": cand.time_ms,
            }
        )

    subgroups_ranked = pd.DataFrame(records_ranked)
    print_header("Top Subgroups")
    print(subgroups_ranked)

    eval_records = []
    coeff_records = []

    for idx, cand in enumerate(top_candidates, start=1):
        cand.summary = getattr(cand, "summary", None)
        eval_result = evaluate_subgroup_models(
            candidate=cand,
            df=df,
            df_attrs=df_attrs,
            feature_config=feature_config,
            global_res=global_res,
            design_df=design_df,
        )
        eval_result["rank"] = idx
        eval_result["rule"] = format_rule(cand.rule)
        eval_result["support"] = cand.support
        eval_result["depth"] = cand.depth
        eval_result["cook_score"] = cand.cook_score
        eval_result["score_type"] = cand.score_type
        eval_records.append(eval_result)

        summary: Optional[SubgroupModelSummary] = eval_result.get("model_summary")  # type: ignore[assignment]
        if summary is None:
            eval_result["R2_Gk"] = np.nan
            eval_result["adj_R2_Gk"] = np.nan
            continue
        eval_result["R2_Gk"] = summary.r2
        eval_result["adj_R2_Gk"] = summary.adj_r2
        for name, beta_val, se_val, beta_global in zip(
            global_res.feature_names,
            summary.beta,
            summary.se,
            global_res.beta,
        ):
            coeff_records.append(
                {
                    "rank": idx,
                    "rule": format_rule(cand.rule),
                    "coef_name": name,
                    "beta_Gk": beta_val,
                    "se_Gk": se_val,
                    "beta_global": beta_global,
                    "beta_diff": beta_val - beta_global,
                }
            )

    subgroup_models_eval = pd.DataFrame(eval_records)
    coeff_summaries = pd.DataFrame(coeff_records)

    if "model_summary" in subgroup_models_eval.columns:
        subgroup_models_eval = subgroup_models_eval.drop(columns=["model_summary"])

    print_header("Subgroup Models Evaluation")
    print(subgroup_models_eval)

    print_header("Coefficient Summaries")
    print(coeff_summaries)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("/mnt/data")
    subgroups_path = output_dir / f"subgroups_ranked_{timestamp}.csv"
    eval_path = output_dir / f"subgroup_models_eval_{timestamp}.csv"
    coeff_path = output_dir / f"coeff_summaries_{timestamp}.csv"

    subgroups_ranked.to_csv(subgroups_path, index=False)
    subgroup_models_eval.to_csv(eval_path, index=False)
    coeff_summaries.to_csv(coeff_path, index=False)

    print_header("Exported CSV Files")
    print(str(subgroups_path))
    print(str(eval_path))
    print(str(coeff_path))

    metric_better_counts = {metric: 0 for metric in SELECTED_METRICS}
    metric_best = {metric: (None, np.inf if BETTER_DIRECTION[metric] == "lower" else -np.inf) for metric in SELECTED_METRICS}
    metric_worst = {metric: (None, -np.inf if BETTER_DIRECTION[metric] == "lower" else np.inf) for metric in SELECTED_METRICS}

    for row in subgroup_models_eval.itertuples():
        for metric in SELECTED_METRICS:
            delta = getattr(row, f"delta_{metric}")
            if pd.isna(delta):
                continue
            better_dir = BETTER_DIRECTION[metric]
            if (better_dir == "lower" and delta < 0) or (better_dir == "higher" and delta > 0):
                metric_better_counts[metric] += 1
            current_best = metric_best[metric]
            current_worst = metric_worst[metric]
            if better_dir == "lower":
                if delta < current_best[1]:
                    metric_best[metric] = (row.rule, delta)
                if delta > current_worst[1]:
                    metric_worst[metric] = (row.rule, delta)
            else:
                if delta > current_best[1]:
                    metric_best[metric] = (row.rule, delta)
                if delta < current_worst[1]:
                    metric_worst[metric] = (row.rule, delta)

    print_header("Metric Improvements Summary")
    for metric in SELECTED_METRICS:
        print(
            f"Subgroups better than global for {metric}: {metric_better_counts[metric]}"
        )
        best_rule, best_delta = metric_best[metric]
        worst_rule, worst_delta = metric_worst[metric]
        if best_rule is None or not np.isfinite(best_delta):
            print(f"Best delta {metric}: None")
        else:
            print(f"Best delta {metric}: {best_delta} (rule: {best_rule})")
        if worst_rule is None or not np.isfinite(worst_delta):
            print(f"Worst delta {metric}: None")
        else:
            print(f"Worst delta {metric}: {worst_delta} (rule: {worst_rule})")


if __name__ == "__main__":
    main()
