#Define train test split
#define variables to test and find subgroups on
import pandas as pd
from sklearn.model_selection import train_test_split
#imports from other files
from regression import (
    train_basic_linear_regression,
    train_complex_linear_regression,
    collect_subgroup_models,
    save_models_csv,
    rebuild_models,
    final_estimator_with_coefs,
    extract_linear_coefs)
from evaluation import evaluate_linear_model, get_rows_subgroup, ensure_dict

# --- Central config (exported so other files can import if needed) ---
NUMERIC_COLS = [
    "total_course_activities",
    "active_minutes",
    "nr_distinct_files_viewed",
    "nr_practice_exams_viewed",
]

ATTR_CONFIG = {
    "sex": "categorical",
    "croho": "categorical",
    "course_repeater": "categorical",
    "ECTS": "categorical",
    "GPA": "numeric",
    "origin": "categorical",
}

X_COLS = [
    "total_attended_labsessions",
    "GPA",
    # "nr_distinct_files_viewed",
    # "nr_practice_exams_viewed",
]

Y_COL = "final_exam"

target_col = 'final_exam'
basic_baseline_cols = ['total_attended_labsessions','GPA']
complex_baseline_cols = ['total_attended_labsessions', 'GPA', 'course_repeater']

#set your variables
datafile = '../data_final.csv'
test_size = 0.4

#Load the data and split it into train/test
#This assumes the data is cleaned and there are no NaNs. The whole section below needs to be revised when Hilde has made cleaned the dataset
df = pd.read_csv(datafile)
df = df.copy()
for c in NUMERIC_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=NUMERIC_COLS).reset_index(drop=True)
df = df.dropna(subset=['GPA', 'ECTS'])
train_df, test_df = train_test_split(df, test_size=test_size, random_state=4)

# Train the basic and complex linear regression baseline on train data
basic_model = train_basic_linear_regression(train_df, basic_baseline_cols)
complex_model = train_complex_linear_regression(train_df, complex_baseline_cols)

#Run the linear regression models found in subgroup_finder.py(using the different slopes for different folks paper)
models = collect_subgroup_models(train_df, X_COLS, Y_COL, ATTR_CONFIG,)
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







