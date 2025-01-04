""" This script will cover:

- Check & compare sales behavior before, during, and after holidays
- Find out any seasonal (Christmas, Easter, etc) purchase behaviors
-Trends of customer behavior during store opening and closing times """

# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import timedelta
from tabulate import tabulate

# Configure logging to display messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EDA_Holiday_Seasonal_Behavior:
    def __init__(self, train_path, test_path):
        try:
            self.train = pd.read_csv(train_path)
            self.test = pd.read_csv(test_path)
            self.train['Date'] = pd.to_datetime(self.train['Date'])
            self.test['Date'] = pd.to_datetime(self.test['Date'])
            self.train['HolidayPeriod'] = 'Before Holiday'
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")

    def analyze_sales_around_holidays(self, holiday_dates, window_days=7):
        try:
            before_sales = []
            during_sales = []
            after_sales = []

            for holiday in holiday_dates:
                before_start = holiday - timedelta(days=window_days)
                before_end = holiday - timedelta(days=1)
                after_start = holiday + timedelta(days=1)
                after_end = holiday + timedelta(days=window_days)

                before_sales.extend(self.train[(self.train['Date'] >= before_start) & (self.train['Date'] <= before_end)]['Sales'])
                during_sales.extend(self.train[self.train['Date'] == holiday]['Sales'])
                after_sales.extend(self.train[(self.train['Date'] >= after_start) & (self.train['Date'] <= after_end)]['Sales'])

            avg_before = sum(before_sales) / len(before_sales) if before_sales else 0
            avg_during = sum(during_sales) / len(during_sales) if during_sales else 0
            avg_after = sum(after_sales) / len(after_sales) if after_sales else 0

            return avg_before, avg_during, avg_after
        except Exception as e:
            logging.error(f"Error analyzing sales around holidays: {e}")
            return 0, 0, 0

    def plot_sales_around_holidays(self):
        try:
            holidays = self.train[self.train['StateHoliday'] != 0]['Date'].unique()
            avg_before, avg_during, avg_after = self.analyze_sales_around_holidays(holidays)

            comparison_df = pd.DataFrame({
                'Period': ['Before Holiday', 'During Holiday', 'After Holiday'],
                'Average Sales': [avg_before, avg_during, avg_after]
            })

            comparison_table = tabulate(comparison_df, headers='keys', tablefmt='grid')
            logging.info("Sales Around Holidays:\n" + comparison_table)

            plt.figure(figsize=(12, 6))
            sns.barplot(data=comparison_df, x='Period', y='Average Sales', palette=['blue', 'red', 'green'])
            plt.title('Average Sales Before, During, and After Holidays')
            plt.xlabel('Holiday Period')
            plt.ylabel('Average Sales')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting sales around holidays: {e}")

    def plot_sales_around_specific_holidays(self):
        try:
            christmas_dates = self.train[self.train['StateHoliday'] == 3]['Date'].unique()
            easter_dates = self.train[self.train['StateHoliday'] == 2]['Date'].unique()

            avg_before_christmas, avg_during_christmas, avg_after_christmas = self.analyze_sales_around_holidays(christmas_dates)
            avg_before_easter, avg_during_easter, avg_after_easter = self.analyze_sales_around_holidays(easter_dates)

            seasonal_comparison_df = pd.DataFrame({
                'Holiday': ['Christmas', 'Christmas', 'Christmas', 'Easter', 'Easter', 'Easter'],
                'Period': ['Before Holiday', 'During Holiday', 'After Holiday', 'Before Holiday', 'During Holiday', 'After Holiday'],
                'Average Sales': [avg_before_christmas, avg_during_christmas, avg_after_christmas, avg_before_easter, avg_during_easter, avg_after_easter]
            })

            seasonal_comparison_table = tabulate(seasonal_comparison_df, headers='keys', tablefmt='grid')
            logging.info("Sales Around Specific Holidays:\n" + seasonal_comparison_table)

            plt.figure(figsize=(14, 8))
            sns.barplot(data=seasonal_comparison_df, x='Holiday', y='Average Sales', hue='Period', palette='viridis')
            plt.title('Average Sales Before, During, and After Holidays')
            plt.xlabel('Holiday')
            plt.ylabel('Average Sales')
            plt.legend(title='Period')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting sales around specific holidays: {e}")

    def plot_customer_behavior(self):
        try:
            self.train['Hour'] = self.train['Date'].dt.hour

            hourly_customers = self.train.groupby('Hour')['Customers'].mean().reset_index()
            hourly_customers_table = tabulate(hourly_customers, headers='keys', tablefmt='grid')
            logging.info("Hourly Customers:\n" + hourly_customers_table)

            plt.figure(figsize=(14, 8))
            sns.lineplot(data=hourly_customers, x='Hour', y='Customers', marker='o')
            plt.title('Average Number of Customers by Hour of the Day', fontsize=20)
            plt.xlabel('Hour of the Day', fontsize=16)
            plt.ylabel('Average Number of Customers', fontsize=16)
            plt.xticks(range(0, 24))
            plt.grid(True)
            plt.show()

            day_of_week_customers = self.train.groupby('DayOfWeek')['Customers'].mean().reset_index()
            day_of_week_customers['DayOfWeek'] = day_of_week_customers['DayOfWeek'].map({
                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
            })
            day_of_week_customers_table = tabulate(day_of_week_customers, headers='keys', tablefmt='grid')
            logging.info("Day of Week Customers:\n" + day_of_week_customers_table)

            monthly_customers = self.train.groupby('Month')['Customers'].mean().reset_index()
            monthly_customers['Month'] = monthly_customers['Month'].map({
                1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
            })
            monthly_customers_table = tabulate(monthly_customers, headers='keys', tablefmt='grid')
            logging.info("Monthly Customers:\n" + monthly_customers_table)

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            sns.barplot(data=day_of_week_customers, x='DayOfWeek', y='Customers', palette='viridis', ax=axes[0])
            axes[0].set_title('Average Number of Customers by Day of the Week', fontsize=20, fontweight='bold')
            axes[0].set_xlabel('Day of the Week', fontsize=16, fontweight='bold')
            axes[0].set_ylabel('Average Number of Customers', fontsize=16, fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45, labelsize=14)
            axes[0].tick_params(axis='y', labelsize=14)
            axes[0].grid(True)

            sns.barplot(data=monthly_customers, x='Month', y='Customers', palette='viridis', ax=axes[1])
            axes[1].set_title('Average Number of Customers by Month', fontsize=20, fontweight='bold')
            axes[1].set_xlabel('Month', fontsize=16, fontweight='bold')
            axes[1].set_ylabel('Average Number of Customers', fontsize=16, fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45, labelsize=14)
            axes[1].tick_params(axis='y', labelsize=14)
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting customer behavior: {e}")

# Example usage:
# eda = EDA_Holiday_Seasonal_Behavior('../src/Data/cleaned_train.csv', '../src/Data/cleaned_test.csv')
# eda.plot_sales_around_holidays()
# eda.plot_sales_around_specific_holidays()
# eda.plot_customer_behavior()