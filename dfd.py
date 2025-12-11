import numpy as np
import random
import pandas as pd
from joblib import Parallel, delayed



def get_distribution_params(dataset_descriptors=None, dataset_targets=None, theta=None, m=None, c=None):
    
    best_m_qms = get_m_false_discoveries(dataset_descriptors=dataset_descriptors, dataset_targets=dataset_targets, theta=theta, m=m, c=c)
    
    # Get distribution params
    mu = np.mean(best_m_qms)
    std = np.std(best_m_qms)
    
    # Get p-values
    p90 = mu + 1.645 * std
    p95 = mu +  1.96 * std
    p99 = mu +  2.58 * std
    
    dfd_params = {'m': m, 'theta': theta, 'c': c, 
                  'qms': best_m_qms, 'mu': mu, 'std': std, 
                  'p90': p90, 'p95': p95, 'p99': p99}
    
    return dfd_params
    

def get_m_false_discoveries(dataset_descriptors=None, dataset_targets=None, theta=None, m=None, c=None):
    
    # Perform process in parallel to increase time efficiency
    best_m_qms = Parallel(n_jobs=3)(delayed(get_false_discovery)(dataset_descriptors=dataset_descriptors, dataset_targets=dataset_targets, theta=theta, c=c) for i in range(m))
    
    return best_m_qms


#########################
# This should be altered to shuffle your descriptors.



def shuffle_descriptors(dataset=None):
    
    # Isolate the PID from the descriptors
    PID_slice = dataset['PID']
    descriptors_slice = dataset.drop('PID', axis=1)
    
    original_index = PID_slice.index 
    
    # Shuffle the rows of the descriptors
    shuffled_descriptors = descriptors_slice.sample(frac = 1).reset_index()
    shuffled_descriptors = shuffled_descriptors.drop('index', axis=1)
    
    # Combine the original PID with  the now shuffled descriptors
    shuffled_descriptors.insert(0, 'PID', PID_slice)
    
    return shuffled_descriptors


def get_false_discovery(dataset_descriptors=None, dataset_targets=None, theta=None, c=None):
    
    # Shuffle the descriptors to remove any correlation between the descriptors and targets
    shuffled_descriptors = shuffle_descriptors(dataset_descriptors)
    
    # Perform beam search on the shuffled dataset
    result_beamsearch = 'Add your Beam Search Here' #pbs.runBS(dataset_descriptors=shuffled_descriptors, dataset_targets=dataset_targets, theta=theta, q=1, c=c) #other BS params stay as default
    best_qm = result_beamsearch[0][1]
    
    return best_qm
