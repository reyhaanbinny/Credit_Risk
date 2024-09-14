# Credit Risk Analysis

Welcome to my **Credit Risk Analysis** project! This project demonstrates a machine learning approach to analyzing credit risk, using a dataset of customer financial and demographic information to predict loan approval decisions.

## Objectives
- Predict whether a credit card application will be approved or denied based on customer information.
- Clean and preprocess the data for optimal model performance.
- Evaluate and compare multiple machine learning models for the best accuracy and performance.

## Project Overview
- **Tools**: Python, Jupyter, Scikit-learn
- **Skills Applied**: Data Cleaning, Feature Engineering, Model Optimization, Classification
- **Models Used**: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting
- **Dataset**: [Download the dataset here](./credit_risk_dataset.csv)

## Key Features
### 1. Data Preprocessing and Cleaning
- Cleaned the dataset by handling missing values and removing approximately **5% of outliers** that could skew the results.
- Performed **data transformation** using a **Column Transformer** to handle categorical and numerical features separately.
- Implemented **Pipelines** to streamline the preprocessing, making it easier to manage and replicate.

### 2. Feature Engineering
- Applied domain knowledge to engineer new features from existing variables.
- Normalized continuous variables and encoded categorical variables using one-hot encoding.
- Reduced dimensionality by dropping irrelevant or highly correlated features.

### 3. Modeling
- Tested multiple classification algorithms, including:
  - **Logistic Regression**
  - **Random Forest Classifier**
  - **Gradient Boosting Classifier**
  - **Support Vector Machines (SVM)**
  - **K-Nearest Neighbors (KNN)**
- Automated model training across multiple classifiers using a unified pipeline, making it easy to compare their performances.
- Used **Randomized Search CV** to tune hyperparameters and optimize the model for each classifier.

### 4. Performance Evaluation
- Achieved a **weighted average F1-score of 0.91**, indicating strong overall performance.
- Visualized model performance using:
  - **Precision-Recall Curve**: To understand how well the model distinguishes between high and low-risk customers.
  - **Learning Curve**: Used to assess bias-variance tradeoffs and detect overfitting or underfitting in the model.

### 5. Model Optimization
- Based on insights from the **Learning Curve**, refined the model to reduce bias and variance, ultimately improving predictive accuracy.
- Selected the best model based on F1-score, precision, recall, and ROC-AUC metrics.

## Tools & Technologies Used
- **Python**: For data analysis, visualization, and modeling.
- **Libraries**: 
  - **Pandas** and **NumPy** for data wrangling and manipulation.
  - **Scikit-learn** for machine learning algorithms and model evaluation.
  - **Matplotlib** and **Seaborn** for data visualization.
- **Jupyter Notebook**: For documenting the process and sharing the results.

## Key Insights
- Features like **annual income**, **credit score**, and **debt-to-income ratio** were identified as key predictors in determining credit risk.
- Customers with a **higher debt-to-income ratio** and **lower credit scores** were more likely to have their credit card applications denied.
- The **Random Forest Classifier** and **Gradient Boosting Classifier** outperformed other models in terms of both precision and recall, making them the best-suited algorithms for this problem.


## Key Visualizations
<br>**Precision-Recall Curve:**<br>
![Precision-Recall Curve](./images/precision_recall_curve.JPG)

<br>

**Learning Curve:**<br>
![Learning Curve](./images/learning_curve.JPG)
<br>

View the full analysis here:  [Credit Risk Analysis](./Credit_Risk.html)

## Project Files
- Download the Jupyter Notebook: [Credit_Risk.ipynb](./Credit_Risk.ipynb)

## Future Improvements
1. **Feature Engineering**:
   - Further refine feature engineering by incorporating external data sources such as credit bureau data to improve predictions.
   - Explore the impact of additional financial indicators like monthly debt payments and existing credit limits.

2. **Model Deployment**:
   - Convert the final model into an API that can be integrated with real-time systems used by financial institutions.
   - Build a web interface that allows users to input customer details and get predictions in real-time.

3. **Deep Learning**:
   - Explore using deep learning techniques such as neural networks to handle complex relationships in the data.
   - Perform feature importance analysis to identify the most significant predictors in credit risk assessment.

## Conclusion
This project demonstrates the ability to predict credit risk using machine learning techniques. The insights derived from this analysis can help financial institutions reduce risk by identifying applicants who are more likely to default on credit obligations. The model can be further improved and integrated into business systems to enhance decision-making processes.


## Contact
Feel free to reach out if you have any questions or feedback!
- **Email**: reyhaanbinny97@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/reyhaanbinny


