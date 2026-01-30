import numpy as np
import random
import pandas as pd
from joblib import Parallel, delayed
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

TARGETS = [
    'G1',
    'G2',
    'absences',
    'activities',
    'paid',
    'famsup',
    'schoolsup',
    'studytime'
]

DESCRIPTORS = [ 
    'school',
    'sex',
    'age',
    'address',
    'famsize',
    'Pstatus',
    'Medu',
    'Fedu',
    'Mjob',
    'Fjob',
    'reason',
    'guardian',
    'traveltime',
    'failures',
    'nursery',
    'higher',
    'internet',
    'romantic',
    'famrel',
    'freetime',
    'goout',
    'Dalc',
    'Walc',
    'health'
 ]


# Define which columns are numeric #None here bc all are in bins
NUMERIC_COLS = [
    'age',
    'Medu',
    'Fedu',
    'traveltime',
    'studytime',
    'failures',
    'famrel',
    'freetime',
    'goout',
    'Dalc',
    'Walc',
    'health',
    'absences',
    'G1',
    'G2',
    'G3'
]

ATTR_CONFIG = {
    key: 'numeric' if key in NUMERIC_COLS else 'categorical' for key in DESCRIPTORS
}


# Define target variable 
Y_COL = "G3"

# Define features used for regression
X_COLS = [
       x for x in TARGETS if x != Y_COL
]


def get_distribution_params(course=None, m=100, test_size=0.3, min_size=20):
    
    best_m_qms = get_m_false_discoveries(course=course, m=m, test_size=test_size, min_size=min_size)
    
    # Get distribution params
    mu = np.mean(best_m_qms)
    std = np.std(best_m_qms)
    
    # Get p-values
    p90 = mu + 1.645 * std
    p95 = mu +  1.96 * std
    p99 = mu +  2.58 * std
    
    dfd_params = {'m': m, 
                  'qms': best_m_qms, 'mu': mu, 'std': std, 
                  'p90': p90, 'p95': p95, 'p99': p99}
    
    return dfd_params
    

def get_m_false_discoveries(course=None, m=None, test_size=0.3, min_size=20):
    
    # Perform process in parallel to increase time efficiency
    
    best_m_qms = []
    # performance_pd = pd.DataFrame()

    for i in range(m):
        qm = get_false_discovery(course=course, m=i, test_size=test_size, min_size=min_size)
        best_m_qms.append(qm)
        # performance_pd = pd.concat([performance_pd, dct_df])
    
    return best_m_qms,


#########################
# This should be altered to shuffle your descriptors.



def shuffle_descriptors(dataset=None):
    
    # Isolate the PID from the descriptors
    PID_slice = dataset['STUDENT ID']
    descriptors_slice = dataset.drop('STUDENT ID', axis=1)
    
    original_index = PID_slice.index 
    
    # Shuffle the rows of the descriptors
    shuffled_descriptors = descriptors_slice.sample(frac = 1).reset_index()
    shuffled_descriptors = shuffled_descriptors.drop('index', axis=1)
    
    # Combine the original PID with  the now shuffled descriptors
    shuffled_descriptors.insert(0, 'STUDENT ID', PID_slice)
    
    return shuffled_descriptors


def get_false_discovery(course=None, m=None, test_size=0.3, min_size=20):

    datafile = 'Data/SecondarySchool/'+str(course)+'_cleaned.csv'
    df = pd.read_csv(datafile)

    DESCS = DESCRIPTORS.copy()
    DESCS.append('STUDENT ID')
    TARGETS = [x for x in list(df.columns) if x not in DESCRIPTORS]

    descriptors = df.copy()[DESCS]
    targets = df.copy()[TARGETS]
        
    # Shuffle the descriptors to remove any correlation between the descriptors and targets
    shuffled_descriptors = shuffle_descriptors(descriptors)
    data_shuffled = pd.merge(shuffled_descriptors, targets, on='STUDENT ID', how='inner')
    
    # Perform beam search on the shuffled dataset
    result_beamsearch = main_for_DFD(df=data_shuffled, m=m, test_size=test_size, min_size=min_size) 
    best_qm = result_beamsearch[0]['cookD']
    #print(best_qm)
    
    return best_qm



def main_for_DFD(df=None, m=None, test_size=0.3, min_size=20):
    # Define target variable and set regression parameters
    Y_COL = 'G3'
    predictor_cols = X_COLS
    # Define size of the test set

    # Define size of the test set
    test_size = test_size

    # Load the data and split it into train/test
    # This assumes the data is cleaned and there are no NaNs. 


    # Load the data and split it into train/test
    # This assumes the data is cleaned and there are no NaNs. 

    df[DESCRIPTORS] = df[DESCRIPTORS].astype(str)
    df[X_COLS] = df[X_COLS].astype(float)
    df[Y_COL] = df[Y_COL].astype(float)
    df = df.copy()
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaNs in numeric columns and specifically in GPA and ECTS
    # No NaNs in the data
    # df = df.dropna(subset=NUMERIC_COLS).reset_index(drop=True)
    # df = df.dropna(subset=['GPA', 'ECTS',])

    train_df, test_df = train_test_split(df, test_size=test_size) # no random state outher 

    # Run the linear regression models found in subgroup_finder.py(using the different slopes for different folks paper)
    models = collect_subgroup_models(train_df, X_COLS, Y_COL, ATTR_CONFIG, beam_width = 30,
                                    max_depth = 3,
                                    min_support = min_size,
                                    top_S = 1)
    
    # model_dict = models[0]
    # description = model_dict.get("description")
    
    # subgroups_train = get_rows_subgroup(models, train_df)
    # subgroups_test  = get_rows_subgroup(models, test_df)

    # # Get subgroup data
    # train_sub = subgroups_train.get(description, pd.DataFrame())
    # test_sub  = subgroups_test.get(description, pd.DataFrame())

    # X, local_complex = train_linear_regression(train_sub, feature_cols=predictor_cols, target_col=target_col)

    # metrics_dict = evaluate_linear_model(local_complex, test_sub, X_COLS, Y_COL)
    # del metrics_dict['y_pred']

    # df_dct = pd.DataFrame(metrics_dict, index=[m])

    return models

