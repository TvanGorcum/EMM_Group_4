#Define train test split
#define variables to test and find subgroups on
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#imports from other files
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
    "bool_practice_exams_viewed"
]
#for subgroup finding
ATTR_CONFIG = {
    "sex": "categorical",
    "croho": "categorical",
    "ECTS": "categorical",
    "GPA": "numeric",
    "origin": "categorical",
    "course_repeater": "categorical",
}
#For regression:
X_COLS = [
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
]

Y_COL = "final_exam"

target_col = 'final_exam'
basic_baseline_cols = ['nr_distinct_files_viewed']
complex_baseline_cols = X_COLS
#set your variables
datafile = '../data_final.csv'
test_size = 0.3

#Load the data and split it into train/test
#This assumes the data is cleaned and there are no NaNs. The whole section below needs to be revised when Hilde has made cleaned the dataset
df = pd.read_csv(datafile)
df = df.copy()
for c in NUMERIC_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=NUMERIC_COLS).reset_index(drop=True)
df = df.dropna(subset=['GPA', 'ECTS',])
train_df, test_df = train_test_split(df, test_size=test_size, random_state=4)
# Train the basic and complex linear regression baseline on train data
basic_model = train_basic_linear_regression(train_df, basic_baseline_cols)
complex_model = train_complex_linear_regression(train_df, complex_baseline_cols)

#Run the linear regression models found in subgroup_finder.py(using the different slopes for different folks paper)
models = collect_subgroup_models(train_df, X_COLS, Y_COL, ATTR_CONFIG)
models_usable = rebuild_models(models)
print(models)
print(f"Collected {len(models)} subgroup models.")
# --- Save to CSV (one row per subgroup-term) ---
save_models_csv(models, "subgroup_linear_models1.csv")
# Optional: quick sanity print
print(f"Exported {len(models)} subgroup models to subgroup_linear_models.csv")
#evaluation metrics for baseline models
metrics_basic = evaluate_linear_model(model = basic_model, df = test_df, X_cols= basic_baseline_cols , y_col= target_col)
print('Basic baseline evaluation metrics:', metrics_basic)
metrics_complex = evaluate_linear_model(model = complex_model, df = test_df, X_cols= complex_baseline_cols , y_col= target_col)
print('Complex baseline evaluation metrics:', metrics_complex)

# ---------- build subgroup masks for both train and test ----------
subgroups_train = get_rows_subgroup(models, train_df)  # {description: df_train_sub}
subgroups_test  = get_rows_subgroup(models, test_df)   # {description: df_test_sub}

results_rows = []

# ---------- per-subgroup: evaluate discovered model + fit/evaluate baselines ----------
for model_dict in models:
    description = model_dict.get("description")
    cookD = model_dict.get("cookD", None)
    n_found = model_dict.get("n", None)

    # subgroup data
    train_sub = subgroups_train.get(description, pd.DataFrame())
    test_sub  = subgroups_test.get(description, pd.DataFrame())

    n_train_sub = len(train_sub)
    n_test_sub  = len(test_sub)

    # ---- (A) Evaluate discovered subgroup model on subgroup test ----
    reg, feats = models_usable[description]

    metrics_discovered_raw = evaluate_linear_model(
        model=reg,
        df=test_sub,
        X_cols=feats,
        y_col=target_col
    )
    row_disc = ensure_dict(metrics_discovered_raw)
    row_disc.update(extract_linear_coefs(reg, feats))
    row_disc.update({
        "model_type": "subgroup_model",
        "description": description,
        "cookD": cookD,
        "n_train": n_train_sub,
        "n_test": n_test_sub,
    })
    results_rows.append(row_disc)

    # ---- (B) Fit + evaluate BASIC baseline on subgroup ----
    # Only fit if we have at least some train rows
    if n_train_sub > 0:
            basic_sg_model = train_basic_linear_regression(train_sub, basic_baseline_cols)
            metrics_basic_sg_raw = evaluate_linear_model(
                model=basic_sg_model,
                df=test_sub,
                X_cols=basic_baseline_cols,
                y_col=target_col
            )
            row_basic = ensure_dict(metrics_basic_sg_raw)
            row_basic.update(extract_linear_coefs(basic_sg_model, basic_baseline_cols))
            row_basic.update({
                "model_type": "subgroup_baseline_basic",
                "description": description,
                "cookD": cookD,
                "n_train": n_train_sub,
                "n_test": n_test_sub
            })
            results_rows.append(row_basic)

    # ---- (C) Fit + evaluate COMPLEX baseline on subgroup ----
    if n_train_sub > 0:
            complex_sg_model = train_complex_linear_regression(train_sub, complex_baseline_cols)
            metrics_complex_sg_raw = evaluate_linear_model(
                model=complex_sg_model,
                df=test_sub,
                X_cols=complex_baseline_cols,
                y_col=target_col
            )
            row_complex = ensure_dict(metrics_complex_sg_raw)
            row_complex.update(extract_linear_coefs(complex_sg_model, complex_baseline_cols))
            row_complex.update({
                "model_type": "subgroup_baseline_complex",
                "description": description,
                "cookD": cookD,
                "n_train": n_train_sub,
                "n_test": n_test_sub,
            })
            results_rows.append(row_complex)

#
mb = ensure_dict(metrics_basic)
mb.update(extract_linear_coefs(basic_model, basic_baseline_cols))
mb.update({
    "model_type": "global_baseline_basic",
    "description": "N/A",
    "cookD": None,
    "n_train": len(train_df),
    "n_test": len(test_df),
})
results_rows.append(mb)

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

#Save
results_df = pd.DataFrame.from_records(results_rows)
results_df.to_csv("subgroup_model_results.csv", index=False)

#Testing approach 2
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

    print("TEST DECISION:", summary_bits)

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





