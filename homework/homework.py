import pandas as pd
import numpy as np
import json
import gzip
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Get project root directory (where homework.py is located)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create directories if they don't exist
os.makedirs(os.path.join(PROJECT_ROOT, 'files', 'models'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'files', 'output'), exist_ok=True)

# Load data using absolute paths
train_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'files', 'input', 'train.csv'))
test_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'files', 'input', 'test.csv'))

# Preprocess data
train_data['Age'] = 2021 - train_data['Year']
test_data['Age'] = 2021 - test_data['Year']

# Split features and target
X_train = train_data.drop(['Selling_Price', 'Year', 'Car_Name'], axis=1)
y_train = train_data['Selling_Price']
X_test = test_data.drop(['Selling_Price', 'Year', 'Car_Name'], axis=1)
y_test = test_data['Selling_Price']

# Define numeric and categorical columns
numeric_features = ['Present_Price', 'Driven_Kms', 'Age', 'Owner']
categorical_features = ['Fuel_Type', 'Selling_Type', 'Transmission']

# Create preprocessing steps
numeric_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse=False)

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(score_func=f_regression)),
    ('regressor', LinearRegression())
])

# Define parameter grid
param_grid = {
    'selector__k': [5, 7, 9, 11]
}

# Create grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Save the model using absolute path
model_path = os.path.join(PROJECT_ROOT, 'files', 'models', 'model.pkl.gz')
with gzip.open(model_path, 'wb') as f:
    pickle.dump(grid_search, f)

# Calculate metrics
def calculate_metrics(y_true, y_pred, dataset_type):
    return {
        'type': 'metrics',
        'dataset': dataset_type,
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mad': mean_absolute_error(y_true, y_pred)
    }

# Get predictions
train_pred = grid_search.predict(X_train)
test_pred = grid_search.predict(X_test)

# Calculate and save metrics
train_metrics = calculate_metrics(y_train, train_pred, 'train')
test_metrics = calculate_metrics(y_test, test_pred, 'test')

# Save metrics using absolute path
metrics_path = os.path.join(PROJECT_ROOT, 'files', 'output', 'metrics.json')
with open(metrics_path, 'w') as f:
    f.write(json.dumps(train_metrics) + '\n')
    f.write(json.dumps(test_metrics) + '\n')
