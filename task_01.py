# Task - 01 : Implement linear regression to predict housing prices based on features like square footage, number of bedrooms, and location. Use a dataset like the Boston Housing dataset for training and evaluation.

#1. Importing Necessary Libraries


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Loading and Exploring the Dataset

# Loading the Dataset
data = pd.read_csv('BostonHousing.csv')

# Display basic dataset information
print("Dataset Info:")
print(data.info())

print("\nFirst 5 rows of the dataset:")
print(data.head())

# 3. Checking for Missing Values and Handling Them

# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# Fill missing values for the 'rm' column if present
if 'rm' in data.columns:
    data['rm'].fillna(data['rm'].mean(), inplace=True)
    print("\nMissing values filled for 'rm' column.")

# Verify remaining missing values
print("\nRemaining missing values:")
print(data.isnull().sum())

# 4.Exploratory Data Analysis (EDA)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Target Variable Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['medv'], kde=True, bins=30)
plt.title("Target Variable Distribution (medv)")
plt.xlabel("Median Value of Owner-Occupied Homes")
plt.show()

# 5. Feature scaling and splitting data

# Features and Target Variable
X = data.drop('medv', axis=1)
y = data['medv']

# Standardize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Training Linear Regression Model

# Train Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
linear_y_pred = linear_model.predict(X_test)

"""## 7.Evaluating the model"""

# Evaluate performance using metrics
linear_mae = mean_absolute_error(y_test, linear_y_pred)
linear_mse = mean_squared_error(y_test, linear_y_pred)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_y_pred)

print("\nLinear Regression Performance:")
print(f"Mean Absolute Error (MAE): {linear_mae:.2f}")
print(f"Mean Squared Error (MSE): {linear_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {linear_rmse:.2f}")
print(f"R² Score: {linear_r2:.2f}")

# Visualizing Results
# Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, linear_y_pred, alpha=0.7, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Perfect Fit')
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()
