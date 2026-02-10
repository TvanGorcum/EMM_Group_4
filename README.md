# Discovering Stratified Learning Strategies for Student Subgroups with Exceptional Study Behavior: More Fine-grained Significance Categories in Regression Models

This repository contains code supporting our Paper "Discovering Stratified Learning Strategies for Student Subgroups with Exceptional Study Behavior: More Finegrained Significance Categories in Regression Models".

## Overview
This code conducts Beam Search Subgroup Discovery on public and private datasets (containing student learning behaviour and performance) and detects false group discovered with the use of linear regression.

## Dataset 
To conduct our tests we use both private and public datasets. The public dataset for mathematics and portuguese course is 10.24432/C5TG7T (DOI). The private datasets will be published in future upon acceptance of the paper

## File structure
```
EMM_Group_4/
├── Data/                             # Folder with public dataset
├── data/                             # Folder with private dataset
├── figures/                          # Generated figures (created at run-time)
├── results/                          # Generated outputs (created at run-time)
├── dfd_ss.py                         # Entry point for distribution-free discovery for portuguese and mathematics public data.
├── Experiment_calculus.py            # Main entry point for calculus private data: split data, train global model, mine/evaluate subgroups, write results
├── Experiment_FDA.py                 # Main entry point for data analytics private data: split data, train global model, mine/evaluate subgroups, write results
├── Experiment_SS.py                  # Main entry point for portuguese and mathematics public data: split data, train global model, mine/evaluate subgroups, write results
├── evaluation.py                     # Metric computation, subgroup mask parsing, helpers (ensure_dict, get_rows_subgroup)
├── inspect_private_calc.ipynb        # Python notebook with the final results for the private calculus data
├── inspect_private_fda.ipynb         # Python notebook with the final results for the private FDA data
├── inspect_private_mat.ipynb         # Python notebook with the final results for the public mathematics data
├── inspect_private_por.ipynb         # Python notebook with the final results for the public portuguese data
├── README.md                         # Project overview and usage instructions
├── regression.py                     # OLS training, extracting coefs/p-values, subgroup term construction, CSV export
├── requirements.txt                  # Pinned Python package versions
└── subgroup_finder.py                # EMM beam search, atomic conditions, numeric binning, Cook’s distance + OLS stats
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
<li>joblib==1.4.2</li>
</ul>

### How to use

#### 1. Clone the repository
```bash
$ git clone <repository_url>
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

#### 4. Run the main script (data file must be located in the ./Data folder)
```bash
$ python Experiment_<name_of_dataset>.py
```
Where `<name_of_dataset>` can be one of:
- `calculus`
- `FDA`
- `SS`

#### 5. Inspect the results
Open and execute one of the notebooks for each dataset:
- **Private Calculus**: `inspect_private_calc.ipynb`
- **Private FDA**: `inspect_private_fda.ipynb`
- **Public Mathematics**: `inspect_private_mat.ipynb`
- **Public Portuguese**: `inspect_private_por.ipynb`

## Authors 
<ul>
<li>Anonymous</li>
</ul>



