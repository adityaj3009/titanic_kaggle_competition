# Titanic - Machine Learning from Disaster

This repository contains a solution to the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic), where the goal is to predict the survival of passengers based on their demographic and travel information.

---

## 📝 Project Overview
The Titanic dataset contains information about the passengers, such as age, sex, ticket class, and more. The objective is to build a machine learning model that predicts whether a passenger survived or not.

---

## 📊 Libraries Used
- **NumPy** (`np`) – for numerical computations  
- **Pandas** (`pd`) – for data manipulation  
- **Seaborn** (`sns`) – for data visualization  
- **Matplotlib.pyplot** (`pyplot`) – for plotting  
- **Scikit-learn** – for machine learning models and evaluation  
  - `train_test_split`  
  - `LogisticRegression`  
  - `RandomForestClassifier`  
  - `accuracy_score`  
  - `GridSearchCV`  

---

## ⚡ Approach
1. **Data Cleaning & Exploration**  
   - Handle missing values (`Age`, `Cabin`)  
   - Encode categorical features (`Sex`, `Embarked`)  
   - Visualize distributions and correlations  

2. **Feature Engineering**  
   - Create new features (if applicable)  
   - Drop irrelevant columns (like `Ticket`, `Name`)  

3. **Model Training & Evaluation**  
   - Split data into training and testing sets  
   - Trained **Logistic Regression** and **Random Forest Classifier**  
   - Evaluated using **accuracy score**  

---

## 🏆 Results
- **Random Forest Classifier with GridSearchCV Accuracy (on Kaggle test set):** 0.72248  
- **CSV Submission Accuracy:** 80.3%  

---

## 📁 Files
- `titanic.ipynb` – Jupyter notebook with all analysis and model training  
- `submission.csv` – CSV file ready for Kaggle submission  

---

## 📈 Visualizations
- Passenger survival distribution by gender, class, and age  
- Correlation heatmaps to identify important features  

---

## 💡 Notes
- Model performance can be improved with hyperparameter tuning, ensemble methods, and advanced feature engineering.
- This project is beginner-friendly and demonstrates a complete ML workflow from data preprocessing to model evaluation.

