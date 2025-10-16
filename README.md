# KNN_MLProject6

💼 Adult Income Prediction using KNN
📚 Project Overview

The goal of this project is to predict whether an individual's annual income exceeds $50K based on demographic, occupational, and educational attributes.

We leverage the K-Nearest Neighbors (KNN) algorithm, a simple yet powerful supervised learning technique that predicts the label of a data point based on the labels of its nearest neighbors in feature space.

This project demonstrates the full machine learning workflow, including data exploration, cleaning, preprocessing, modeling, and evaluation.

🗂 Dataset


Total Records: 48,842

Total Features: 15

Key Features
Feature	Type	Description
age	numeric	Age of the individual
workclass	categorical	Type of employment
education	categorical	Highest education level
marital-status	categorical	Marital status
occupation	categorical	Job type
relationship	categorical	Relationship status
race	categorical	Race
gender	categorical	Gender
capital-gain	numeric	Income from capital gains
capital-loss	numeric	Income from capital losses
hours-per-week	numeric	Number of working hours per week
native-country	categorical	Country of origin
income	categorical	Target variable (<=50K or >50K)
🛠 Steps Followed
1️⃣ Data Loading

Dataset loaded using Pandas.

Verified shape, columns, data types, and missing values.

2️⃣ Data Exploration

Statistical summary for numeric features using describe().

Value counts and uniqueness for categorical features to understand distributions.

Checked for duplicate rows.

3️⃣ Data Cleaning

Removed duplicates.

Replaced "?" entries with NaN in workclass, occupation, and native-country.

Dropped rows with missing values (less than 6% of the dataset).

Simplified marital status:

Married-civ-spouse & Married-AF-spouse → Married

Never-married → Single

Other → Not-Married

Encoded gender: Male=1, Female=0.

Removed fnlwgt column (low predictive value).

4️⃣ Data Preprocessing

Feature-target split:

X = df.drop('income', axis=1)

y = df['income']

Train-test split: 80% training, 20% testing using train_test_split.

5️⃣ Feature Scaling & Encoding

Numeric features (age, capital-gain, capital-loss, hours-per-week) → RobustScaler for scaling.

Categorical features:

education → OrdinalEncoder based on educational hierarchy.

workclass, occupation, native-country → BinaryEncoder (reduces high cardinality).

marital-status, relationship, race → OneHotEncoder (drop first to avoid dummy trap).

Target variable (income) → LabelEncoder (<=50K=0, >50K=1).

6️⃣ Model Training

Algorithm: K-Nearest Neighbors (KNN)

Parameters: n_neighbors=15

The model learns patterns by comparing each test point to its closest neighbors in feature space.

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

7️⃣ Model Evaluation

Metrics Used:

Accuracy

Confusion Matrix

Classification Report (precision, recall, F1-score)

Results:

Metric	Value
Train Accuracy	72.23%
Test Accuracy	72.45%

Confusion Matrix:

	Pred <=50K	Pred >50K
True <=50K	4989	1853
True >50K	636	1557

Insights:

Model performs better for majority class (<=50K).

Minority class (>50K) predictions are moderate (precision = 0.46, recall = 0.71).

🔍 Theoretical Explanation

KNN works based on the principle of similarity:

Each data point is placed in a feature space.

For a new observation, KNN finds the k nearest neighbors and predicts the class by majority voting.

Advantages: Simple, non-parametric, interpretable.

Limitations: Sensitive to feature scaling, high-dimensionality, and imbalanced datasets.

Why KNN is chosen here:

The dataset has mixed numeric and categorical features.

KNN is a baseline model to understand data separability.

Works well for small to medium-sized datasets.

🎯 Conclusion

Achieved ~72% test accuracy.

Model predicts majority class well but needs improvement for minority class (>50K).

Preprocessing (scaling, encoding) is critical for KNN performance.

🚀 Future Improvements

Handle class imbalance using SMOTE or undersampling.

Try tree-based models (Random Forest, XGBoost) for better handling of categorical features.

Hyperparameter tuning for KNN (n_neighbors) and distance metrics.

Feature engineering for high-cardinality categorical variables.

📊 Visualization

Histograms for numeric features.

Correlation heatmap to identify feature relationships.

Value counts for categorical variables.

📝 References

Scikit-learn documentation for KNN

Pandas and Plotly for data processing and visualization.
