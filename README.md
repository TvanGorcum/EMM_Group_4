# EMM Regression Framework

This repository contains a lightweight framework that applies Exceptional Model Mining (EMM) as a preprocessing step to enhance linear regression models. The workflow mines subgroups on the training split using a fixed beam-search strategy, converts those subgroups into indicator features, and evaluates whether the augmented feature set improves predictive performance.

## Framework structure

```
emm_framework/
├── __init__.py
├── emm.py            # Beam search implementation
├── models.py         # Linear regression helpers and metrics
├── pipeline.py       # Orchestrates the full workflow
└── schema.py         # Predicate and schema primitives
```

An example script demonstrating end-to-end usage on the included dummy dataset lives in `examples/run_pipeline.py`.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage example

```python
import pandas as pd
from emm_framework import (
    AttributeSpec,
    EMMRegressionPipeline,
    SubgroupSchema,
)

# Load pre-cleaned data
df = pd.read_csv("student_dummy_data.csv")
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

schema = SubgroupSchema.from_attributes(
    [
        AttributeSpec.numeric("num_clicks", [(0, 20), (20, 40), (40, None)]),
        AttributeSpec.numeric("previous_grade", [(0, 70), (70, 85), (85, None)]),
        AttributeSpec.numeric("homework_score", [(0, 70), (70, 85), (85, None)]),
    ]
)

pipeline = EMMRegressionPipeline(schema)
result = pipeline.run(
    train_frame=train_df,
    test_frame=test_df,
    target="exam_grade",
    features=["num_clicks", "previous_grade", "homework_score"],
)

print("Baseline test RMSE:", result.baseline_test.rmse)
print("Augmented test RMSE:", result.augmented_test.rmse)
for subgroup in result.subgroups:
    print(subgroup.describe())
```

## Tests

Run the demonstration script:

```bash
python examples/run_pipeline.py
```
