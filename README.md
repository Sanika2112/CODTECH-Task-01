# CODTECH-Task-01

**Name**: SANIKA SHINDE<br>
**Company**: CODTECH IT SOLUTIONS<br>
**ID**: CT6WDS2686<br>
**Domain**: Machine Learning<br>
**Duration**: DECEMBER 5th, 2024 to JANUARY 20th, 2025<br>

## Overview of project
### Topic: Housing Price Prediction using Linear Regression
This project implements a linear regression model to predict housing prices using the Boston Housing Dataset. It leverages features like square footage, number of rooms, pollution levels, and socio-economic factors to estimate the median value of owner-occupied homes.

The project includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation to demonstrate the application of machine learning in real-world scenarios.
---
### Dataset
The dataset used is the Boston Housing Dataset, which contains:<br>

**Features**: 13 variables representing socio-economic, environmental, and structural characteristics.<br>
**Target Variable**: medv (Median value of owner-occupied homes in $1000s).<br>
---
### **Workflow**
1. **Import Libraries**:
   Load necessary Python libraries such as `pandas`, `sklearn`, and `matplotlib`.
2. **Data Loading**:
   Load the dataset into a `pandas` DataFrame.
3. **Data Cleaning**:
   - Handle missing values by imputing with the column mean.
   - Identify and remove outliers if required.
4. **Exploratory Data Analysis**:
   - Generate a **correlation heatmap** to understand relationships between variables.
   - Visualize the distribution of the target variable (`medv`).
5. **Data Splitting**:
   Divide the dataset into **training (80%)** and **testing (20%)** subsets.
6. **Model Training**:
   - Train a linear regression model on the training data.
   - Use features like `rm` (number of rooms) and `lstat` (lower status population).
7. **Evaluation**:
   - Calculate performance metrics such as:
     - **Mean Absolute Error (MAE)**.
     - **R² Score**.
   - Plot actual vs. predicted values to visualize the model's performance.

---

### **Key Features**
- **Correlation Heatmap**: Visualizes the relationships between variables to identify influential predictors.
- **Model Performance Metrics**: Evaluate predictions using MAE and R² Score.
- **Data Visualization**: Includes target variable distribution and prediction scatterplots.

---

### **Requirements**
The following Python libraries are required:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---
### Result
![image](https://github.com/user-attachments/assets/59a33c94-6fd6-40b7-8a17-afdad54178f2)
