# Rossmann Pharmaceuticals Sales Forecasting

## Business Need
Forecast sales in all stores across several cities six weeks ahead of time using machine learning models.

## Data and Features
- Promotions
- Competition
- School and state holidays
- Seasonality
- Locality

## Learning Outcomes
- Exploration of customer purchasing behavior
- Prediction of store sales
- Machine learning approach
- Deep Learning approach
- Serving predictions on a web interface

## Competency Mapping
- Data preprocessing
- Model building
- Model evaluation
- Model serving

## Team
- Machine Learning Engineers
- Data Scientists
- Finance Analysts

## Key Dates
- Project kickoff: [Date]
- Data exploration completion: [Date]
- Model training completion: [Date]
- API deployment: [Date]

## Instructions
### Task 1 - Exploration of Customer Purchasing Behavior
- Clean data and handle outliers/missing data
- Visualize features and interactions
- Analyze customer behavior with respect to promos, holidays, seasonality, etc.

### Task 2 - Prediction of Store Sales
#### 2.1 Preprocessing
- Convert non-numeric columns to numeric
- Handle NaN values
- Generate new features from datetime columns
- Scale data using standard scaler

#### 2.2 Building Models with Sklearn Pipelines
- Use tree-based algorithms (e.g., Random Forest Regressor)
- Work with sklearn pipelines for modular and reproducible modeling

#### 2.3 Choose a Loss Function
- Select and defend an appropriate loss function for the regression problem

#### 2.4 Post Prediction Analysis
- Explore feature importance
- Estimate confidence intervals for predictions

#### 2.5 Serialize Models
- Save models with timestamps for tracking predictions

#### 2.6 Building Model with Deep Learning
- Create an LSTM model using TensorFlow or PyTorch
- Ensure the model is not too deep (two layers)
- Preprocess time series data and build LSTM regression model

### Task 3 - Model Serving API Call
- Create a REST API using Flask
- Load the trained model and scaler
- Define API endpoints for predictions
- Handle requests and preprocess input data
- Return predictions as JSON response
- Deploy API to a web server or cloud platform

---
