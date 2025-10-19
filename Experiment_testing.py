import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Imports from other files
from regression import (
    train_basic_linear_regression,
    train_complex_linear_regression,
    collect_subgroup_models,
    save_models_csv,
    rebuild_models,
    final_estimator_with_coefs,
    extract_linear_coefs,
    add_subgroup_terms,
    partial_f_test,
    _design_matrix,
    _ols_with_stats_matrix,
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
    "logged_in_weekly",
    "nr_files_viewed",
    "nr_slides_viewed",
    "nr_practice_exams_viewed",
    #"bool_practice_exams_viewed"
]

# Define target variable and set regression parameters
Y_COL = "CalculatedNumericResult"
target_col = 'CalculatedNumericResult'
complex_baseline_cols = X_COLS
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
complex_model = train_complex_linear_regression(train_df, complex_baseline_cols)

# Run the linear regression models found in subgroup_finder.py(using the different slopes for different folks paper)
models = collect_subgroup_models(train_df, X_COLS, Y_COL, ATTR_CONFIG)
models_usable = rebuild_models(models)
#print(models)
print(f"Collected {len(models)} subgroup models.")
# Save to CSV (one row per subgroup-term)
save_models_csv(models, "subgroup_linear_models1.csv")
print(f"Exported {len(models)} subgroup models to subgroup_linear_models.csv")

# Evaluation metrics for baseline model
metrics_complex = evaluate_linear_model(model = complex_model, df = test_df, X_cols= complex_baseline_cols , y_col= target_col)
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
        model=complex_model,
        df=test_sub,
        X_cols=complex_baseline_cols,
        y_col=target_col,
    )
    row_global_on_sub = ensure_dict(metrics_global_on_sub)
    row_global_on_sub.update(extract_linear_coefs(complex_model, complex_baseline_cols))
    row_global_on_sub.update({
        "model_type": "subgroup_baseline_global",
        "description": description,
        "cookD": round(cookD, 5),
        "n_train": n_train_sub,
        "n_test": n_test_sub,
    })
    results_rows.append(row_global_on_sub)

    # Retrain the same global architecture on subgroup train and evaluate on subgroup test
    # Only do this when we have at least one training row in the subgroup
    if n_train_sub > 0:
        local_complex = train_complex_linear_regression(train_sub, complex_baseline_cols)
        metrics_local_complex = evaluate_linear_model(
            model=local_complex,
            df=test_sub,
            X_cols=complex_baseline_cols,
            y_col=target_col,
        )
        row_local = ensure_dict(metrics_local_complex)
        row_local.update(extract_linear_coefs(local_complex, complex_baseline_cols))
        row_local.update({
            "model_type": "subgroup_model",  # retrained-on-subgroup row
            "description": description,
            "cookD": round(cookD, 5),
            "n_train": n_train_sub,
            "n_test": n_test_sub,
        })
        results_rows.append(row_local)


mc = ensure_dict(metrics_complex)
mc.update(extract_linear_coefs(complex_model, complex_baseline_cols))
mc.update({
    "model_type": "global_baseline_complex",
    "description": "N/A",
    "cookD": None,
    "n_train": len(train_df),
    "n_test": len(test_df),
})
results_rows.append(mc)

# Save all results of the fitted subgroups
results_df = pd.DataFrame.from_records(results_rows)
results_df.to_csv("subgroup_model_results.csv", index=False)

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
global_model = train_basic_linear_regression(train_df, feature_cols=current_feature_cols, target_col=target_col)

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
    if test_aug[gamma_name].sum() == 0 or train_aug[gamma_name].sum() == 0:
        print(f"[Skip] '{desc}': subgroup has no rows in train or test.")
        continue
    y_test = test_aug[target_col].to_numpy()
    Xr_test, names_r = _design_matrix(test_aug, reduced_cols, add_intercept=True)
    Xf_test, names_f = _design_matrix(test_aug, full_cols, add_intercept=True)
    try:
        F, pF, q, df_full = partial_f_test(y_test, Xr_test, Xf_test)
    except Exception as e:
        print(f"[Skip] '{desc}': F-test failed ({e}).")
        continue
    fit_full_test = _ols_with_stats_matrix(Xf_test, y_test)
    added_term_indices = [names_f.index(c) for c in added_cols if c in names_f]
    added_t = fit_full_test["t"][added_term_indices]
    added_p = fit_full_test["p"][added_term_indices]
    # new: pick significant terms only
    pmap = {col: float(fit_full_test["p"][names_f.index(col)]) for col in added_cols}
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
        "kept_cols": significant_cols,  # <-- add this for transparency
    }

    #print("TEST DECISION:", summary_bits)

    if keep:
        # only add the significant columns to the model
        current_feature_cols = reduced_cols + significant_cols

        # fit the train model on exactly those columns
        global_model = train_basic_linear_regression(
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
pd.DataFrame(kept_rows).to_csv("kept_subgroups_testing_phase.csv", index=False)





