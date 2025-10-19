import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm

# Imports from other files
from regression import (
    train_linear_regression,
    collect_subgroup_models,
    save_models_csv,
    extract_linear_coefs,
    add_subgroup_terms,
    _augment_with_kept,)
from evaluation import evaluate_linear_model, get_rows_subgroup, ensure_dict

# Define which columns are numeric
NUMERIC_COLS = [
    "total_attended_labsessions",
    "active_minutes",
    "nr_distinct_files_viewed",
    "total_course_activities",
    "nightly_activities",
    "distinct_days",
    "logged_in_weekly",
    "nr_files_viewed",
    "nr_slides_viewed",
    "nr_practice_exams_viewed",
    #"bool_practice_exams_viewed"
]

# Define attributes for subgroup discovery
ATTR_CONFIG = {
    "sex": "categorical",
    "croho": "categorical",
    "ECTS": "categorical",
    "GPA": "numeric",
    "origin": "categorical",
    "course_repeater": "categorical",
}

# Define features used for regression
X_COLS = [
    "total_attended_labsessions",
    "active_minutes",
    "nr_distinct_files_viewed",
    "total_course_activities",
    #"nightly_activities",
    "distinct_days",
    #"logged_in_weekly",
    "nr_files_viewed",
    #"nr_slides_viewed",
    "nr_practice_exams_viewed",
    #"bool_practice_exams_viewed"
]

# Define target variable and set regression parameters
Y_COL = "CalculatedNumericResult"
target_col = 'CalculatedNumericResult'
predictor_cols = X_COLS
datafile = '../data_final.csv'

# Define size of the test set
test_size = 0.4

# Load the data and split it into train/test
# This assumes the data is cleaned and there are no NaNs. 

df = pd.read_csv(datafile)
df = df.copy()
for c in NUMERIC_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with NaNs in numeric columns and specifically in GPA and ECTS
df = df.dropna(subset=NUMERIC_COLS).reset_index(drop=True)
df = df.dropna(subset=['GPA', 'ECTS',])

train_df, test_df = train_test_split(df, test_size=test_size, random_state=4)

# Train the global linear regression on all train data
global_model = train_linear_regression(train_df, predictor_cols)

# Run the linear regression models found in subgroup_finder.py(using the different slopes for different folks paper)
models = collect_subgroup_models(train_df, X_COLS, Y_COL, ATTR_CONFIG)
#print(models)
print(f"Collected {len(models)} subgroup models.")
# Save to CSV (one row per subgroup-term)
save_models_csv(models, "results/subgroup_linear_models.csv")
print(f"Exported {len(models)} subgroup models to results/subgroup_linear_models.csv")

# Evaluation metrics for baseline model
metrics_complex = evaluate_linear_model(model = global_model, df = test_df, X_cols= predictor_cols , y_col= target_col)
print('Complex baseline evaluation metrics:', metrics_complex)

# Build subgroup masks for both train and test
subgroups_train = get_rows_subgroup(models, train_df)
subgroups_test  = get_rows_subgroup(models, test_df)

results_rows = []


#
#
#
# Start of approach 1: evaluate each subgroup model individually
#
# For each subgroup that was found, compare the residuals on the subgroup test set from the global model
# with the residuals on the subgroup test set after retraining the same model architecture only on the subgroup train set.
#
#
#

# Per-subgroup: evaluate discovered model and baseline
for model_dict in models:
    description = model_dict.get("description")
    cookD = model_dict.get("cookD", None)
    n_found = model_dict.get("n", None)

    # Get subgroup data
    train_sub = subgroups_train.get(description, pd.DataFrame())
    test_sub  = subgroups_test.get(description, pd.DataFrame())

    n_train_sub = len(train_sub)
    n_test_sub  = len(test_sub)

    # If there's no test data for this subgroup we can't evaluate — skip
    if n_test_sub == 0:
        continue

    # Evaluate global model on this subgroup's test set (subgroup baseline)
    metrics_global_on_sub = evaluate_linear_model(
        model=global_model,
        df=test_sub,
        X_cols=predictor_cols,
        y_col=target_col,
    )
    row_global_on_sub = ensure_dict(metrics_global_on_sub)
    # Add coefficients and p-values for each coefficient
    row_global_on_sub.update(extract_linear_coefs(global_model, predictor_cols))
    row_global_on_sub.update({
        "model_type": "subgroup_global_baseline",
        "description": description,
        "cookD": cookD,
        "n_train": n_train_sub,
        "n_test": n_test_sub,
    })
    results_rows.append(row_global_on_sub)

    # Retrain the same global architecture on subgroup train and evaluate on subgroup test
    # Only do this when we have at least one training row in the subgroup
    if n_train_sub > 0:
        local_complex = train_linear_regression(train_sub, predictor_cols)
        metrics_local_complex = evaluate_linear_model(
            model=local_complex,
            df=test_sub,
            X_cols=predictor_cols,
            y_col=target_col,
        )
        row_local = ensure_dict(metrics_local_complex)
        # Add coefficients and p-values for each coefficient
        row_local.update(extract_linear_coefs(local_complex, predictor_cols))
        row_local.update({
            "model_type": "subgroup_model",
            "description": description,
            "cookD": round(cookD, 5),
            "n_train": n_train_sub,
            "n_test": n_test_sub,
        })
        results_rows.append(row_local)


mc = ensure_dict(metrics_complex)
mc.update(extract_linear_coefs(global_model, predictor_cols))
mc.update({
    "model_type": "global",
    "description": "N/A",
    "cookD": None,
    "n_train": len(train_df),
    "n_test": len(test_df),
})
results_rows.append(mc)

# Save all results of the fitted subgroups
results_df = pd.DataFrame.from_records(results_rows)
results_df.to_csv("results/subgroup_model_results.csv", index=False)

#
#
#
# Start of approach 2: Add significant subgroup terms to global model
#
# 1) Start with the global model and evaluate its fit on the full test set
# 2) For each subgroup that was found, in descending order of interestingness (Cook's Distance):
# 3)    Add an interaction variable for each regressor with inclusion in the subgroup
# 4)    For each of the added variables, perform a t-test to see if this variable has added explainatory power on the test set
# 5)        If it has, then we keep it, if it has not, we do not use it.
#
#
#

ALPHA_F = 0.05
ALPHA_T = 0.05
BASE_GLOBAL_COLS = X_COLS[:]
KEPT_SUBGROUPS = []

models_sorted = sorted(models, key=lambda m: m.get("cookD", -np.inf), reverse=True)
current_feature_cols = BASE_GLOBAL_COLS[:]
global_model = train_linear_regression(train_df, feature_cols=current_feature_cols, target_col=target_col)

for m in models_sorted:
    desc = m["description"]
    # Build frames that include ALL previously kept subgroup terms
    train_aug_all = _augment_with_kept(train_df, KEPT_SUBGROUPS, BASE_GLOBAL_COLS)
    test_aug_all = _augment_with_kept(test_df, KEPT_SUBGROUPS, BASE_GLOBAL_COLS)

    # Now add the CURRENT candidate subgroup on top
    train_aug, gamma_name, inter_cols = add_subgroup_terms(train_aug_all, desc, BASE_GLOBAL_COLS)
    test_aug, _, _ = add_subgroup_terms(test_aug_all, desc, BASE_GLOBAL_COLS, gamma_name=gamma_name)

    reduced_cols = current_feature_cols[:]  # includes earlier kept gammas & interactions
    added_cols = [gamma_name] + inter_cols  # current candidate's new terms
    full_cols = reduced_cols + added_cols

    # Skip if subgroup has no rows in train or test
    if test_aug[gamma_name].sum() == 0 or train_aug[gamma_name].sum() == 0:
        print(f"[Skip] '{desc}': subgroup has no rows in train or test.")
        continue

    # Prepare design matrices with intercept
    Xr_test = sm.add_constant(test_aug[reduced_cols], has_constant='add')
    Xf_test = sm.add_constant(test_aug[full_cols], has_constant='add')
    y_test = test_aug[target_col].values

    # Fit reduced and full models on test set
    model_reduced = sm.OLS(y_test, Xr_test).fit()
    model_full = sm.OLS(y_test, Xf_test).fit()

    # Partial F-test
    rss_r = sum(model_reduced.resid ** 2)
    rss_f = sum(model_full.resid ** 2)
    df_f = model_full.df_resid
    q = Xf_test.shape[1] - Xr_test.shape[1]
    F = ((rss_r - rss_f) / q) / (rss_f / df_f)
    try:
        from scipy.stats import f as fdist
        pF = fdist.sf(F, q, df_f)
    except Exception:
        pF = np.nan

    # t-tests for added terms
    added_term_indices = [i for i, c in enumerate(Xf_test.columns) if c in added_cols]
    added_t = model_full.tvalues.iloc[added_term_indices]
    added_p = model_full.pvalues.iloc[added_term_indices]
    pmap = {col: float(model_full.pvalues.get(col, np.nan)) for col in added_cols}
    significant_cols = [col for col in added_cols if pmap[col] < ALPHA_T]

    keep = (pF < ALPHA_F) and (len(significant_cols) > 0)

    summary_bits = {
        "description": desc,
        "cookD": m.get("cookD"),
        "F_stat": float(F),
        "F_pvalue": float(pF) if pF == pF else None,
        "t_added": [float(t) if t == t else None for t in np.atleast_1d(added_t)],
        "p_added": [float(p) if p == p else None for p in np.atleast_1d(added_p)],
        "kept": bool(keep),
        "kept_cols": significant_cols,
    }

    if keep:
        # only add the significant columns to the model
        current_feature_cols = reduced_cols + significant_cols

        # fit the train model on exactly those columns
        global_model = train_linear_regression(
            train_aug, feature_cols=current_feature_cols, target_col=target_col
        )

        # store only the kept interactions (gamma may or may not be kept)
        kept_inters = [c for c in inter_cols if c in significant_cols]
        KEPT_SUBGROUPS.append((desc, gamma_name, kept_inters, summary_bits))

print(f"\n== FINAL MODEL FEATURES ({len(current_feature_cols)}): {current_feature_cols}")
print(f"Kept {len(KEPT_SUBGROUPS)} subgroups (by F- & t-tests on hold-out).")

kept_rows = []
for (desc, gamma_name, inter_cols, summ) in KEPT_SUBGROUPS:
    kept_rows.append({
        "description": desc,
        "gamma": gamma_name,
        "interaction_cols": "|".join(inter_cols),
        **{k: v for k, v in summ.items() if k not in ("description",)},
    })
pd.DataFrame(kept_rows).to_csv("results/kept_subgroups_testing_phase.csv", index=False)





