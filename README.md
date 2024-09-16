# Credit Risk Analysis

## Project Overview
This project focuses on predicting **credit card approval** using a **classification model** based on customer financial and demographic data. The goal is to accurately classify applicants into either "approved" or "denied" categories, helping financial institutions manage credit risk efficiently.

## Objective
The key objectives of this project are:
- To clean, process, and analyze a dataset of over **32,000 customer records**.
- To apply **machine learning classification models** to predict credit card approval.
- To evaluate model performance and fine-tune it for maximum accuracy.

## Key Features
1. **Data Preprocessing**:
   - Handled missing values and removed **5% outliers**.
   - Standardized features using **ColumnTransformer** and optimized preprocessing through a **Pipeline**.

2. **Modeling & Optimization**:
   - Implemented multiple classifiers, including **Logistic Regression**, **Random Forest**, and **XGBoost**.
   - Used **RandomizedSearchCV** for hyperparameter tuning.
   - Achieved a weighted **F1-score** of **0.91**.

3. **Evaluation Metrics**:
   - Precision-Recall and **ROC Curves**.
   - Learning curves to check for **bias-variance tradeoff** and **overfitting**.

## Tools & Technologies
- **Python**: Data analysis and machine learning.
- **Pandas & NumPy**: Data preprocessing and manipulation.
- **Scikit-learn**: Model building and evaluation.
- **Matplotlib & Seaborn**: Data visualization.

## How to Run
1. Clone the repository.
2. Run the Jupyter notebook or Python scripts.

