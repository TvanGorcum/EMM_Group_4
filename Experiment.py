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
    NUMERIC_COLS, X_COLS, Y_COL, ATTR_CONFIG,)
from evaluation import evaluate_linear_model, evaluate_models_by_subgroup

target_col = 'CalculatedNumericResult'
basic_baseline_cols = ['ECTS', 'GPA']
complex_baseline_cols = ['ECTS', 'GPA', 'course_repeater']

#set your variables
datafile = '../data_final.csv'
test_size = 0.2

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
basic_model = train_basic_linear_regression(train_df)
complex_model = train_complex_linear_regression(train_df)

#Run the linear regression models found in subgroup_finder.py(using the different slopes for different folks paper)
models = collect_subgroup_models(train_df,)
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





