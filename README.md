# Deposit Classification using Stacking Ensemble

This project implements a **classification model** to predict whether a customer will make a bank deposit, using **stacking ensemble methods** combining XGBoost and Random Forest classifiers, with GradientBoostingClassifier as meta-learner.

## Features

- **Data preprocessing** and cleaning
- **Exploratory Data Analysis (EDA)** with visualization
- **Feature engineering** and encoding
- **Modeling** using:
  - XGBoost
  - Random Forest
  - Stacking ensemble with GradientBoostingClassifier as meta-learner
- **Main Model evaluation**:
  - ROC AUC

## Dataset

The dataset is a **bank marketing dataset** for deposit prediction, containing customer demographic and financial attributes.

Example columns:
- Age
- Job
- Marital status
- Education
- Default
- Housing loan
- Personal loan
- Contact type
- Campaign
- Deposit (Target Variable)

## Usage

1. Clone this repository
2. Open `deposit-classification.ipynb` with Jupyter Notebook or Jupyter Lab
3. Run all cells to execute the pipeline
4. Inspect the model performance

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install the requirements with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Example Code Snippet

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('training_dataset.csv')
print(df.head())
```

## Results

- The stacking ensemble model achieved around **88% accuracy** 
- ROC AUC model achieved around **76%**
