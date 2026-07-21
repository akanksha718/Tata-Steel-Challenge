# Defect Detection in Hot Rolling

## Problem Overview
In Hot Rolling Mills, one specific defect (referred to here as the **Alpha defect**) is a critical quality challenge. This defect cannot be detected through the existing system because the coil remains under tension in the inspection zones. 

Since it is not possible to detect Alpha defects inline, current quality control relies on sample observations at the final stage, where only a certain percentage of the total coils produced are inspected. Additionally, manual inspection is time-intensive, whereas the manufacturing and supply chain processes operate under strict time constraints. Although the Alpha defect accounts for only a very small percentage of the total production volume, it can still lead to customer complaints and product downgrades.

## Task
During hot rolling, each stage has different process parameters that can contribute to the formation of the defect. Therefore, all stages must be considered to effectively detect the formation of Alpha defects.

Detect the occurrence of the Alpha defect during rolling to prevent customer complaints and reduce downgrades through proactive action.

## Dataset Description
The dataset contains process parameters collected across multiple manufacturing stages:

| File | Dimensions | Description |
| :--- | :--- | :--- |
| `train.csv` | 1352 × 51 | Training data with features and target label |
| `test.csv` | 339 × 50 | Test data with process parameters |
| `sample_submission.csv` | 339 × 2 | Format for expected predictions |

### Variable Description
* **`CoilID`**: Unique identifier for each coil.
* **`X1` – `X49`**: Process parameters across multiple stages.
* **`Y`**: Target variable indicating Alpha defect occurrence (`1` = Defect, `0` = No Defect).

## Evaluation Metric
A model which will have **0 false negatives** and **less than 10% false positives** will be accepted:
* **Recall**: 100%
* **Precision**: > 90%

## Submission Criteria
* The submission file must be submitted in `.csv` format.
* The size of the submission file must be **339 × 2**.
* Ensure the submission file contains:
  * Correct `CoilID` values matching the test file.
  * Correct column names as provided in `sample_submission.csv`.