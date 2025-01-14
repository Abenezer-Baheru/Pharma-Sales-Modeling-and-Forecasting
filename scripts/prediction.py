import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SalesPrediction:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train = None
        self.test = None
        self.holidays = None
        self.pipeline = None

    def load_data(self):
        """Load the training and test datasets."""
        try:
            self.train = pd.read_csv(self.train_path)
            self.test = pd.read_csv(self.test_path)
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")

    def fill_na_values(self, df):
        """Fill missing values in the dataset."""
        try:
            df['StateHoliday'].fillna(0, inplace=True)
            df['CompetitionDistance'].fillna(df['CompetitionDistance'].mean(), inplace=True)
            df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
            df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
            df['Promo2SinceWeek'].fillna(0, inplace=True)
            df['Promo2SinceYear'].fillna(0, inplace=True)
            df['PromoInterval'].fillna('None', inplace=True)
            logging.info("NaN values filled successfully.")
        except Exception as e:
            logging.error(f"Error filling NaN values: {e}")
        return df

    def extract_date_features(self, df):
        """Extract additional features from the date column."""
        try:
            df['Weekday'] = pd.to_datetime(df['Date']).dt.weekday
            df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
            df['Is_Beginning_Month'] = df['Day'].apply(lambda x: 1 if x <= 10 else 0)
            df['Is_Mid_Month'] = df['Day'].apply(lambda x: 1 if 10 < x <= 20 else 0)
            df['Is_End_Month'] = df['Day'].apply(lambda x: 1 if x > 20 else 0)
            logging.info("Additional features extracted successfully.")
        except Exception as e:
            logging.error(f"Error extracting additional features: {e}")
        return df

    def calculate_holiday_features(self, df):
        """Calculate the number of days to the next holiday and the number of days after the last holiday."""
        try:
            def days_to_next_holiday(date, holidays):
                future_holidays = [holiday for holiday in holidays if holiday >= date]
                if future_holidays:
                    return (future_holidays[0] - date).days
                else:
                    return np.nan

            def days_after_last_holiday(date, holidays):
                past_holidays = [holiday for holiday in holidays if holiday <= date]
                if past_holidays:
                    return (date - past_holidays[-1]).days
                else:
                    return np.nan

            df['Days_To_Next_Holiday'] = df['Date'].apply(lambda x: days_to_next_holiday(pd.to_datetime(x), self.holidays))
            df['Days_After_Last_Holiday'] = df['Date'].apply(lambda x: days_after_last_holiday(pd.to_datetime(x), self.holidays))
            df['Days_To_Next_Holiday'].fillna(0, inplace=True)
            logging.info("Holiday features calculated successfully.")
        except Exception as e:
            logging.error(f"Error calculating holiday features: {e}")
        return df

    def encode_features(self, df):
        """Encode categorical features using one-hot encoding."""
        try:
            non_numeric_columns = ['StoreType', 'Assortment', 'PromoInterval']
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_features = encoder.fit_transform(df[non_numeric_columns])
            encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(non_numeric_columns))
            df = pd.concat([df, encoded_df], axis=1)
            storetype_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
            assortment_mapping = {'a': 1, 'b': 2, 'c': 3}
            df['StoreType'] = df['StoreType'].map(storetype_mapping)
            df['Assortment'] = df['Assortment'].map(assortment_mapping)
            df = df.drop(columns=['PromoInterval'])
            logging.info("Features encoded successfully.")
        except Exception as e:
            logging.error(f"Error encoding features: {e}")
        return df

    def scale_data(self, train, test):
        """Scale numeric features using StandardScaler."""
        try:
            numeric_columns_train = ['Sales', 'Customers', 'StateHoliday', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day', 'WeekOfYear', 'Weekday', 'Days_To_Next_Holiday', 'Days_After_Last_Holiday', 'StoreType_b', 'StoreType_c', 'StoreType_d', 'Assortment_b', 'Assortment_c', 'PromoInterval_Jan,Apr,Jul,Oct', 'PromoInterval_Mar,Jun,Sept,Dec', 'PromoInterval_None']
            numeric_columns_test = ['StateHoliday', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day', 'WeekOfYear', 'Weekday', 'Days_To_Next_Holiday', 'Days_After_Last_Holiday', 'StoreType_b', 'StoreType_c', 'StoreType_d', 'Assortment_b', 'Assortment_c', 'PromoInterval_Jan,Apr,Jul,Oct', 'PromoInterval_Mar,Jun,Sept,Dec', 'PromoInterval_None']
            scaler_train = StandardScaler()
            scaler_test = StandardScaler()
            train_scaled = train.copy()
            train_scaled[numeric_columns_train] = scaler_train.fit_transform(train[numeric_columns_train])
            test_scaled = test.copy()
            test_scaled[numeric_columns_test] = scaler_test.fit_transform(test[numeric_columns_test])
            logging.info("Data scaled successfully.")
        except Exception as e:
            logging.error(f"Error scaling data: {e}")
        return train_scaled, test_scaled

    def save_data(self, train_scaled, test_scaled):
        """Save the scaled data to CSV files."""
        try:
            train_scaled.to_csv('../src/Data/scaled_ML_train.csv', index=False)
            test_scaled.to_csv('../src/Data/scaled_ML_test.csv', index=False)
            logging.info("Data scaled and saved successfully.")
        except Exception as e:
            logging.error(f"Error saving data: {e}")

    def train_model(self, train_scaled, test_scaled):
        """Train the RandomForestRegressor model and evaluate its performance."""
        try:
            target = 'Sales'
            features = train_scaled.columns.drop([target, 'Date'])
            X_train, X_val, y_train, y_val = train_test_split(train_scaled[features], train_scaled[target], test_size=0.2, random_state=42)
            self.pipeline = Pipeline([
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            self.pipeline.fit(X_train, y_train)
            y_pred = self.pipeline.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            logging.info(f'Mean Squared Error: {mse}')
            logging.info(f'Mean Absolute Error: {mae}')
            logging.info(f'R-squared: {r2}')
            self.plot_results(y_val, y_pred, mse, mae, r2)
            self.plot_feature_importances(X_train)
            self.calculate_confidence_interval(y_pred)
            self.save_model()
        except Exception as e:
            logging.error(f"Error training model: {e}")

        def plot_results(self, y_val, y_pred, mse, mae, r2):
            """Plot the actual vs predicted values and display evaluation metrics."""
            try:
                mse_rounded = round(mse, 4)
                mae_rounded = round(mae, 4)
                r2_rounded = round(r2, 4)
                plt.figure(figsize=(14, 7))
                plt.plot(y_val.values, label='Actual', color='blue')
                plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
                plt.xlabel('Data Points')
                plt.ylabel('Sales')
                plt.title('Actual vs Predicted Sales')
                plt.legend()
                plt.text(0.05, 0.95, f'MSE: {mse_rounded}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                plt.text(0.05, 0.90, f'MAE: {mae_rounded}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                plt.text(0.05, 0.85, f'R²: {r2_rounded}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                plt.show()
                logging.info("Results plotted successfully.")
            except Exception as e:
                logging.error(f"Error plotting results: {e}")
    def save_model(self):
        """Save the trained model to a file with a timestamp."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            joblib.dump(self.pipeline, f'./model_{timestamp}.pkl')
            logging.info(f'Model saved as model_{timestamp}.pkl')
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def run(self):
        """Run the entire data processing and model training pipeline."""
        self.load_data()
        if self.train is not None and self.test is not None:
            self.train = self.fill_na_values(self.train)
            self.test = self.fill_na_values(self.test)
            self.train = self.extract_date_features(self.train)
            self.test = self.extract_date_features(self.test)
            self.holidays = pd.to_datetime(self.train[self.train['StateHoliday'] != 0]['Date'].unique())
            self.train = self.calculate_holiday_features(self.train)
            self.test = self.calculate_holiday_features(self.test)
            self.train = self.encode_features(self.train)
            self.test = self.encode_features(self.test)
            train_scaled, test_scaled = self.scale_data(self.train, self.test)
            self.save_data(train_scaled, test_scaled)
            self.train_model()

if __name__ == "__main__":
    train_path = '../src/Data/cleaned_train.csv'
    test_path = '../src/Data/cleaned_test.csv'
    sales_prediction = SalesPrediction(train_path, test_path)
    sales_prediction.run()