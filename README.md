# Amazon Sales Data Analysis & Modeling

This project analyzes Amazon sales performance using Python-based data exploration, machine learning models, and a simple PyTorch neural network for regression.

## Project Files

- `amazon-dataset-manipulation.ipynb`: Main notebook containing all data cleaning, visualization, and modeling steps.
- `Amazon Sale Report.csv`: Raw sales data containing information on order ID, fulfillment method, product category, shipping, and more.

## Key Features

### Data Preprocessing
- Removed null and duplicate records
- Extracted date components (year, month, day)
- Encoded categorical variables using `LabelEncoder`
- Standardized numerical fields with `StandardScaler`

### Exploratory Data Analysis
- Distribution plots for quantity and amount
- Temporal sales trends (month/day/year)
- Top categories, channels, and fulfillment types visualized using `Seaborn`

### Machine Learning Models
Tested regression models to predict a numerical target from the dataset:
- Linear Regression
- Decision Tree Regressor
- Ridge & Lasso Regression
- Gradient Boosting Regressor
- XGBoost
- Random Forest

Performance was evaluated using **Mean Squared Error (MSE)**.

### Deep Learning with PyTorch
A fully connected neural network (3 layers) was trained to predict target values. Performance visualized with actual vs. predicted plots.

```python
class SimpleNN(nn.Module):
    def __init__(self):
        ...

 
