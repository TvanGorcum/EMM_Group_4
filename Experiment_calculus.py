import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel

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

# course_code,collegeyear,FinalExam,WeeklyTests,StartTest,Midterm,CalculatedNumericResult,EndResult,
# Passed,origin,sex,croho,type_vooropleiding,double_major,course_repeater,active_minutes,total_course_activities,
# nightly_activities,distinct_days,logged_in_weekly,nr_files_viewed,nr_distinct_files_viewed,nr_slides_viewed,nr_practice_exams_viewed

# Define attributes for subgroup discovery
ATTR_CONFIG = {
    "sex": "categorical",
    "croho": "categorical",
    "origin": "categorical",
    "course_repeater": "categorical",
    #"collegeyear": "categorical",
    #"course_code": "categorical",
    "type_vooropleiding" : "categorical",
    "double_major": "categorical",
}

# Define features used for regression
X_COLS = [
    #"active_minutes",
    "nr_distinct_files_viewed",
    "total_course_activities",
    "nightly_activities",
    "distinct_days",
    "logged_in_weekly",
    "nr_files_viewed",
    "nr_slides_viewed",
    #"nr_practice_exams_viewed",
    #"bool_practice_exams_viewed"
]

# Define target variable 
Y_COL = "CalculatedNumericResult"

def approach_one(models, subgroups_train, subgroups_test, global_model, train_df, test_df, predictor_cols, target_col, results_rows, coefs_dict):
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

        # Only do this when we have at least one training row in the subgroup
        if n_train_sub > 0:
            X, local_complex = train_linear_regression(train_sub, predictor_cols, 'CalculatedNumericResult')
            metrics_local_complex = evaluate_linear_model(
                model=local_complex,
                df=test_sub,
                X_cols=predictor_cols,
                y_col=target_col,
                
            )
            
            for col, coef in zip(X.columns, local_complex.params):
                coefs_dict['subgroup'].append('#'+str(i))
                coefs_dict['term'].append(col)
                coefs_dict['coef'].append(coef)
            
            coefs_dict['p'] += list(local_complex.pvalues)
            coefs_dict['significant'] += ['yes' if i <= 0.10 else 'no' for i in local_complex.pvalues ]

            # Calculate residuals for paired tests (keep predictions internal only)
            resid_global = test_sub[target_col].values - metrics_global_on_sub["y_pred"]
            resid_local = test_sub[target_col].values - metrics_local_complex["y_pred"]

            # Use squared residuals (MSE per-sample) instead of absolute residuals (MAE)
            sq_resid_global = resid_global ** 2
            sq_resid_local = resid_local ** 2

            # Also compute a simple mean predictor (mean from the subgroup train set)
            mean_pred = train_sub[target_col].mean()
            resid_mean = test_sub[target_col].values - mean_pred
            sq_resid_mean = resid_mean ** 2
            
            # Calculate mean predictor metrics
            mean_pred_mae = float(np.mean(np.abs(resid_mean)))
            mean_pred_mse = float(np.mean(sq_resid_mean))
            ss_res_mean = np.sum(sq_resid_mean)
            ss_tot_sub = np.sum((test_sub[target_col].values - test_sub[target_col].mean()) ** 2)
            mean_pred_r2 = float(1 - (ss_res_mean / ss_tot_sub)) if ss_tot_sub > 0 else 0.0
            mean_pred_mean_residual = float(np.mean(resid_mean))

            row_local = ensure_dict(metrics_local_complex)
            # Remove y_pred from the row before saving
            if "y_pred" in row_local:
                del row_local["y_pred"]

            row_local.update(extract_linear_coefs(local_complex, predictor_cols))
            row_local.update({
                "model_type": "subgroup",
                "description": description,
                "cookD": cookD,
                "n_train": n_train_sub,
                "n_test": n_test_sub,
                # Global model metrics on this subgroup
                "global_baseline_r2": metrics_global_on_sub["r2"],
                "global_baseline_mae": metrics_global_on_sub["mae"],
                "global_baseline_mse": metrics_global_on_sub["mse"],
                "global_baseline_mean_residual": metrics_global_on_sub["mean_residual"],
                # Mean predictor metrics for this subgroup
                "mean_baseline_r2": mean_pred_r2,
                "mean_baseline_mae": mean_pred_mae,
                "mean_baseline_mse": mean_pred_mse,
                "mean_baseline_mean_residual": mean_pred_mean_residual,
            })

            # Consider only pairs without NaNs for local vs global (MSE)
            mask_g = ~np.isnan(sq_resid_global) & ~np.isnan(sq_resid_local)
            if mask_g.sum() >= 2:
                try:
                    t_stat_g, p_one_g = ttest_rel(sq_resid_local[mask_g],
                                                sq_resid_global[mask_g],
                                                alternative="less")
                    if np.isnan(t_stat_g) or np.isnan(p_one_g):
                        t_stat_g, p_one_g = None, None
                    # Wilcoxon for local vs global (on squared errors)
                    w_stat_g, w_p_g = wilcoxon(sq_resid_local[mask_g], sq_resid_global[mask_g], alternative="less")
                    if np.isnan(w_stat_g) or np.isnan(w_p_g):
                        w_stat_g, w_p_g = None, None
                except Exception:
                    t_stat_g, p_one_g = None, None
                    w_stat_g, w_p_g = None, None
            else:
                t_stat_g, p_one_g = None, None
                w_stat_g, w_p_g = None, None

            # Consider only pairs without NaNs for local vs mean (subgroup mean) using MSE
            mask_m = ~np.isnan(sq_resid_mean) & ~np.isnan(sq_resid_local)
            if mask_m.sum() >= 2:
                try:
                    t_stat_m, p_one_m = ttest_rel(sq_resid_local[mask_m], sq_resid_mean[mask_m], alternative="less")
                    if np.isnan(t_stat_m) or np.isnan(p_one_m):
                        t_stat_m, p_one_m = None, None
                    # Wilcoxon for local vs subgroup-mean (squared errors)
                    w_stat_m, w_p_m = wilcoxon(sq_resid_local[mask_m], sq_resid_mean[mask_m], alternative="less")
                    if np.isnan(w_stat_m) or np.isnan(w_p_m):
                        w_stat_m, w_p_m = None, None
                except Exception:
                    t_stat_m, p_one_m = None, None
                    w_stat_m, w_p_m = None, None
            else:
                t_stat_m, p_one_m = None, None
                w_stat_m, w_p_m = None, None

            # Compare GLOBAL model (on subgroup test set) vs SUBGROUP MEAN predictor using MSE
            mask_mg = ~np.isnan(sq_resid_mean) & ~np.isnan(sq_resid_global)
            if mask_mg.sum() >= 2:
                try:
                    # paired t-test: is global-model MSE < subgroup-mean MSE?
                    t_stat_mg, p_one_mg = ttest_rel(sq_resid_global[mask_mg], sq_resid_mean[mask_mg], alternative="less")
                    if np.isnan(t_stat_mg) or np.isnan(p_one_mg):
                        t_stat_mg, p_one_mg = None, None
                    # Wilcoxon for global vs subgroup-mean (squared errors)
                    w_stat_mg, w_p_mg = wilcoxon(sq_resid_global[mask_mg], sq_resid_mean[mask_mg], alternative="less")
                    if np.isnan(w_stat_mg) or np.isnan(w_p_mg):
                        w_stat_mg, w_p_mg = None, None
                except Exception:
                    t_stat_mg, p_one_mg = None, None
                    w_stat_mg, w_p_mg = None, None
            else:
                t_stat_mg, p_one_mg = None, None
                w_stat_mg, w_p_mg = None, None

            row_local["ttest_p"] = p_one_g
            row_local["ttest_stat"] = t_stat_g
            row_local["wilcoxon_p"] = w_p_g
            row_local["wilcoxon_stat"] = w_stat_g

            row_local["ttest_p_mean"] = p_one_m
            row_local["ttest_stat_mean"] = t_stat_m
            row_local["wilcoxon_p_mean"] = w_p_m
            row_local["wilcoxon_stat_mean"] = w_stat_m

            row_local["ttest_p_mean_global"] = p_one_mg
            row_local["ttest_stat_mean_global"] = t_stat_mg
            row_local["wilcoxon_p_mean_global"] = w_p_mg
            row_local["wilcoxon_stat_mean_global"] = w_stat_mg

            results_rows.append(row_local)
    
    return coefs_dict


def main():
    # Define target variable and set regression parameters
    target_col = 'CalculatedNumericResult'
    predictor_cols = X_COLS
    datafile = 'data/Calc_filtered.csv'
    # Define size of the test set
    test_size = 0.2
    coefs_dict = {'subgroup': [], 'term': [], 'coef': [], 'p': [], 'significant': [] }

    # Load the data and split it into train/test
    # This assumes the data is cleaned and there are no NaNs. 

    df = pd.read_csv(datafile)
    df = df.copy()
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaNs in numeric columns and specifically in GPA and ECTS
    df = df.dropna(subset=NUMERIC_COLS).reset_index(drop=True)
    #df = df.dropna(subset=['GPA', 'ECTS',])

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=4)

    # Train the global linear regression on all train data
    X, global_model = train_linear_regression(train_df, predictor_cols, 'CalculatedNumericResult')

    for col, coef in zip(X.columns, global_model.params):
        coefs_dict['subgroup'].append('Ω')
        coefs_dict['term'].append(col)
        coefs_dict['coef'].append(coef)

    coefs_dict['p'] += list(global_model.pvalues)
    coefs_dict['significant'] += ['yes' if i <= 0.10 else 'no' for i in global_model.pvalues ]

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

    coefs_dict = approach_one(models, subgroups_train, subgroups_test, global_model, train_df, test_df, predictor_cols, target_col, results_rows)

    # Calculate baseline (mean predictor) metrics for global model on full test set
    mean_pred_global = train_df[target_col].mean()
    resid_mean_global = test_df[target_col].values - mean_pred_global
    baseline_mse_global = float(np.mean(resid_mean_global ** 2))
    baseline_mae_global = float(np.mean(np.abs(resid_mean_global)))
    ss_res_baseline = np.sum(resid_mean_global ** 2)
    ss_tot = np.sum((test_df[target_col].values - test_df[target_col].mean()) ** 2)
    baseline_r2_global = float(1 - (ss_res_baseline / ss_tot)) if ss_tot > 0 else 0.0
    baseline_mean_residual_global = float(np.mean(resid_mean_global))

    mc = ensure_dict(metrics_complex)
    # Remove y_pred from the row before saving
    if "y_pred" in mc:
        del mc["y_pred"]
        
    mc.update(extract_linear_coefs(global_model, predictor_cols))
    mc.update({
        "model_type": "global",
        "description": "N/A",
        "cookD": None,
        "n_train": len(train_df),
        "n_test": len(test_df),
        # Add baseline (mean predictor) metrics
        "mean_baseline_r2": baseline_r2_global,
        "mean_baseline_mae": baseline_mae_global,
        "mean_baseline_mse": baseline_mse_global,
        "mean_baseline_mean_residual": baseline_mean_residual_global,
    })
    results_rows.append(mc)

    # Save all results of the fitted subgroups
    results_df = pd.DataFrame.from_records(results_rows)
    results_df.to_csv("results/subgroup_model_results.csv", index=False)

    coefs_df = pd.DataFrame.from_dict(coefs_dict)
    coefs_df.to_csv(f"results/coefs_calculus.csv", index=False)
    
if __name__ == "__main__":
    main()

