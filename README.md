# Identifying Personalized Learning Strategies for Student Subgroups with Exceptional Study Efforts

This repository contains code supporting our Paper "Identifying Personalized Learning Strategies for Student Subgroups with Exceptional Study Efforts".

## Overview 
This code conducts Beam Search Subgroup Discovery on University Learn Management Sytems dataset (containing student learning behaviour and performance) and detects false group discovered with the use of linear regression.

## Dataset 
The dataset contains sensitive information and due to this fact won't be published and widely available.

## File structure
```
EMM_Group_4/
├── Experiment.py                     # Main entry point: split data, train global model, mine/evaluate subgroups, write results
├── evaluation.py                     # Metric computation, subgroup mask parsing, helpers (ensure_dict, get_rows_subgroup)
├── regression.py                     # OLS training, extracting coefs/p-values, subgroup term construction, CSV export
├── subgroup_finder.py                # EMM beam search, atomic conditions, numeric binning, Cook’s distance + OLS stats
├── final_preprocessing.ipynb         # Data cleaning: remove 'ONG', fix comma-decimals, derive bool_practice_exams_viewed; saves 
├── results_inspector.ipynb           # Utilities to inspect results/subgroup_model_results.csv and summarize significance
├── requirements.txt                  # Pinned Python package versions
├── README.md                         # Project overview and usage instructions
└── results/                          # Generated outputs (created at run-time)
    ├── subgroup_model_results.csv    # Per-subgroup metrics and significance tests vs baselines
    └── subgroup_linear_models.csv    # Long-form coef tables for subgroup/global models
```



## Prerequisites and Usage
We tested and implemented this project with **python 3.12.3.** 
The libraries we used and the versions we tested in (also presented in the requirements.txt file):
<ul>
<li>pandas==2.2.3</li>
<li>scipy==1.14.1</li>
<li>numpy==2.0.2</li>
<li>statsmodels==0.14.5</li>
<li>scikit-learn==1.5.2</li>
</ul>

### How to use

#### 1. Clone the repository
```bash
$ git clone https://github.com/TvanGorcum/EMM_Group_4.git
```

#### 2. Initialize and activate local environment
```bash
$ python -m venv path/to/the/environment
$ source path/to/the/environment/bin/activate #Linux/MacOS
$ .\path\to\the\environment\scripts\activate #Windows
```

#### 3. Install required libraries
```bash
$ pip install -r requirements.txt
```

#### 4. Run the main script (data file must be located one directory above the current one)
```bash
$ python Experiments.py
```

## Authors 
<ul>
<li>Teun Van Gorcum</li>
<li>Dani Chambon</li>
<li>Hilde Storms</li>
<li>Neha Bogavarapu</li>
<li>Jan Ludwicki</li>
</ul>



