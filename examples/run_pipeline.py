"""Run the EMM regression pipeline on the dummy student dataset."""

from __future__ import annotations

import pandas as pd

from emm_framework import AttributeSpec, EMMRegressionPipeline, SubgroupSchema


def main() -> None:
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

    print("Baseline train metrics:", result.baseline_train)
    print("Baseline test metrics:", result.baseline_test)
    print("Augmented train metrics:", result.augmented_train)
    print("Augmented test metrics:", result.augmented_test)
    print("\nTop subgroups:")
    for subgroup in result.subgroups:
        print(subgroup.describe())


if __name__ == "__main__":
    main()
