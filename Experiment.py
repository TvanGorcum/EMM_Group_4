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
    rebuild_models,)
from evaluation import evaluate_linear_model, get_rows_subgroup

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

#For each subgroup regression model, find results:
subgroups_dfs = get_rows_subgroup(models, test_df) #Returns a dictionary of dataframes with as key the description of the subgroups and as df the rows which adhere to that subgroup
print(models_usable)
print(subgroups_dfs)
results_sg_models = []
for description, (reg, feats) in models_usable.items():
    subdf = subgroups_dfs[description]
    result_sg_model = evaluate_linear_model(
        model=reg,                 # <- only the sklearn estimator
        df=subdf,
        X_cols=feats,             # <- use the matching feature list
        y_col=target_col
    )
    print(description+':', result_sg_model)
    result_sg_model['description'] = description
    result_sg_model['n_rows_tested'] = len(subdf)
    results_sg_models.append(result_sg_model)

metrics_basic['description'] = 'basic_baseline'
metrics_basic['n_rows_tested'] = len(test_df)
metrics_complex['description'] = 'complex_baseline'
metrics_complex['n_rows_tested'] = len(test_df)
# Add both baselines to the results list
results_sg_models.extend([metrics_basic, metrics_complex])
results_df = pd.DataFrame(results_sg_models)

results_df.to_csv('subgroup_model_results.csv', index=False)






