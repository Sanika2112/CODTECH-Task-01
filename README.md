# CODTECH-Task-01

**Name**: SANIKA SHINDE<br>
**Company**: CODTECH IT SOLUTIONS<br>
**ID**: CT6WDS2686<br>
**Domain**: Machine Learning<br>
**Duration**: DECEMBER 5th, 2024 to JANUARY 20th, 2025<br>

## Project Overview
This project focuses on predicting housing prices using the Boston Housing Dataset by implementing a linear regression model. The aim is to explore the relationship between socio-economic, environmental, and structural factors with housing prices and build an effective predictive model.

---

## Dataset

### Boston Housing Dataset
- **Features**: 13 variables including socio-economic, environmental, and structural characteristics.
- **Target Variable**: `medv` (Median value of owner-occupied homes in $1000s).

---

## Workflow

### 1. Importing Libraries
The following Python libraries are used in this project:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `matplotlib` and `seaborn`: For data visualization.
- `scikit-learn`: For machine learning implementation.

### 2. Data Loading
The dataset is loaded into a `pandas` DataFrame for preprocessing and analysis.

### 3. Data Cleaning
- **Missing Values**: Imputed with the column mean.
- **Outlier Detection**: Outliers are identified and removed as necessary to improve model accuracy.

### 4. Exploratory Data Analysis (EDA)
- **Correlation Heatmap**: Used to visualize the relationships between variables and identify important predictors.
- **Distribution Analysis**: The target variable (`medv`) is visualized to understand its spread and skewness.

### 5. Data Splitting
The dataset is divided into two subsets:
- **Training Set (80%)**: Used to train the linear regression model.
- **Testing Set (20%)**: Used to evaluate model performance.

### 6. Model Training
- A linear regression model is trained using features such as `rm` (number of rooms) and `lstat` (lower status population).

### 7. Evaluation
- **Performance Metrics**:
  - Mean Absolute Error (MAE)
  - R² Score
- **Visualization**:
  - Scatterplot of actual vs. predicted values to assess prediction accuracy.

---

## Key Features
- **Correlation Heatmap**: Identifies significant predictors of housing prices.
- **Evaluation Metrics**: Quantifies model performance using MAE and R².
- **Data Visualization**: Includes a variety of visualizations to understand data and results.

---

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- The following Python libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/housing-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd housing-price-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Project
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run the notebook file `Task_01.ipynb` step by step.
---
### Result
![image](https://github.com/user-attachments/assets/59a33c94-6fd6-40b7-8a17-afdad54178f2)
