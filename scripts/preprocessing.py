import pandas as pd
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, train_path, test_path, store_path):
        self.train_path = train_path
        self.test_path = test_path
        self.store_path = store_path
        self.train = None
        self.test = None
        self.store = None

    def load_data(self):
        try:
            logging.info("Loading datasets...")
            self.train = pd.read_csv(self.train_path)
            self.test = pd.read_csv(self.test_path)
            self.store = pd.read_csv(self.store_path)

            logging.info("Displaying dataset information...")
            logging.info(f"Train dataset size: {self.train.shape}")
            logging.info(f"Test dataset size: {self.test.shape}")
            logging.info(f"Store dataset size: {self.store.shape}")

            logging.info("Train dataset head:\n" + str(self.train.head()))
            logging.info("Test dataset head:\n" + str(self.test.head()))
            logging.info("Store dataset head:\n" + str(self.store.head()))

            logging.info("Train dataset description:\n" + str(self.train.describe()))
            logging.info("Test dataset description:\n" + str(self.test.describe()))
            logging.info("Store dataset description:\n" + str(self.store.describe()))

            logging.info("Train dataset info:\n" + str(self.train.info()))
            logging.info("Test dataset info:\n" + str(self.test.info()))
            logging.info("Store dataset info:\n" + str(self.store.info()))
        except Exception as e:
            logging.error(f"Error loading data: {e}")

    def merge_data(self):
        try:
            logging.info("Merging train and store datasets...")
            self.train = pd.merge(self.train, self.store, on='Store', how='left')
            self.test = pd.merge(self.test, self.store, on='Store', how='left')
        except Exception as e:
            logging.error(f"Error merging data: {e}")

    def visualize_missing_values(self):
        def missing_values_table(df):
            # Total missing values
            mis_val = df.isnull().sum()
            
            # Percentage of missing values
            mis_val_percent = 100 * mis_val / len(df)
            
            # Make a table with the results
            mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
            
            # Rename the columns
            mis_val_table_ren_columns = mis_val_table.rename(
            columns = {0 : 'Missing Values', 1 : '% of Total Values'})
            
            # Sort the table by percentage of missing descending
            mis_val_table_ren_columns = mis_val_table_ren_columns[
                mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
            
            # Print some summary information
            print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")
            
            # Return the dataframe with missing information
            return mis_val_table_ren_columns

        print("Missing values in train dataset:\n", missing_values_table(self.train))
        print("Missing values in test dataset:\n", missing_values_table(self.test))
        print("Missing values in store dataset:\n", missing_values_table(self.store))

    def visualize_outliers(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.train['Sales'])
        plt.title('Box plot of Sales')
        plt.show()

    def count_outliers(self):
        Q1 = self.train['Sales'].quantile(0.25)
        Q3 = self.train['Sales'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.train[(self.train['Sales'] < lower_bound) | (self.train['Sales'] > upper_bound)]
        return outliers.shape[0]

    def clean_data(self):
        try:
            logging.info("Handling white spaces...")
            self.train.columns = self.train.columns.str.strip()
            self.test.columns = self.test.columns.str.strip()
            self.store.columns = self.store.columns.str.strip()

            logging.info("Eliminating duplicates...")
            self.train.drop_duplicates(inplace=True)
            self.test.drop_duplicates(inplace=True)
            self.store.drop_duplicates(inplace=True)

            logging.info("Checking data types...")
            logging.info(self.train.dtypes)
            logging.info(self.test.dtypes)
            logging.info(self.store.dtypes)

            logging.info("Handling missing values...")
            # Fill missing values in 'Open' column of test dataset with mode
            open_mode = self.test['Open'].mode()[0]
            self.test['Open'].fillna(open_mode, inplace=True)

            logging.info("Handling outliers...")
            # Example: Removing outliers in Sales column using IQR method
            Q1 = self.train['Sales'].quantile(0.25)
            Q3 = self.train['Sales'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.train[(self.train['Sales'] < lower_bound) | (self.train['Sales'] > upper_bound)]
            logging.info(f"Number of outliers to be removed: {outliers.shape[0]}")
            # Uncomment the next line to actually remove the outliers
            # self.train = self.train[(self.train['Sales'] >= lower_bound) & (self.train['Sales'] <= upper_bound)]

            logging.info("Converting categorical variables to numeric...")
            self.train['StateHoliday'] = self.train['StateHoliday'].map({'a': 1, 'b': 2, 'c': 3, '0': 0})
            self.test['StateHoliday'] = self.test['StateHoliday'].map({'a': 1, 'b': 2, 'c': 3, '0': 0})
            self.store['StoreType'] = self.store['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
            self.store['Assortment'] = self.store['Assortment'].map({'a': 1, 'b': 2, 'c': 3})

            logging.info("Converting date columns to datetime...")
            self.train['Date'] = pd.to_datetime(self.train['Date'])
            self.test['Date'] = pd.to_datetime(self.test['Date'])

            logging.info("Extracting additional features from date...")
            self.train['Year'] = self.train['Date'].dt.year
            self.train['Month'] = self.train['Date'].dt.month
            self.train['Day'] = self.train['Date'].dt.day
            self.train['WeekOfYear'] = self.train['Date'].dt.isocalendar().week

            self.test['Year'] = self.test['Date'].dt.year
            self.test['Month'] = self.test['Date'].dt.month
            self.test['Day'] = self.test['Date'].dt.day
            self.test['WeekOfYear'] = self.test['Date'].dt.isocalendar().week
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")

    def save_cleaned_data(self, train_output_path, test_output_path):
        try:
            logging.info("Saving cleaned data...")
            self.train.to_csv(train_output_path, index=False)
            self.test.to_csv(test_output_path, index=False)
        except Exception as e:
            logging.error(f"Error saving cleaned data: {e}")