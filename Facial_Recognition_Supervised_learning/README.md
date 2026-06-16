# Facial Recognition using Supervised Learning

## Overview

This project uses supervised machine learning techniques to distinguish images of Arnold Schwarzenegger from images of other individuals.

The dataset is derived from the Labeled Faces in the Wild (LFW) dataset and consists of PCA-transformed facial features. Multiple classification models are evaluated using cross-validation and hyperparameter tuning to determine the best-performing model.

## Objectives

- Build machine learning pipelines for multiple classification algorithms.
- Tune hyperparameters using GridSearchCV.
- Compare model performance using cross-validation.
- Select the best model based on validation performance.
- Evaluate the final model on a held-out test set.

## Models Evaluated

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

## Dataset

The dataset contains PCA-transformed facial image features.

Target labels:

- `1` → Arnold Schwarzenegger
- `0` → Other Individuals

## Results

### Best Model

**Logistic Regression**

### Performance


| Metric                    | Score   |
| ------------------------- | ------- |
| Cross Validation Accuracy | 80.95%  |
| Test Accuracy             | 81.58%  |
| Precision                 | 100.00% |
| Recall                    | 12.50%  |
| F1 Score                  | 22.22%  |


### Notes

The dataset is imbalanced, containing significantly fewer Arnold Schwarzenegger images than images of other individuals. As a result, the selected model achieves strong overall accuracy but lower recall for the positive class.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt

```

## Running the Project

```bash
python main.py

```

## Output

The script:

- Trains and evaluates multiple classification models
- Performs hyperparameter optimization with GridSearchCV
- Prints evaluation metrics
- Generates a confusion matrix visualization

## Example Output



## Future Improvements

- ROC-AUC analysis
- Additional feature engineering
- Model persistence using Joblib
- Interactive Streamlit interface
- Testing on larger facial recognition datasets

