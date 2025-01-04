import unittest
import pandas as pd
from EDA_Holiday_Seasonal_Behavior import EDA_Holiday_Seasonal_Behavior

class TestEDA_Holiday_Seasonal_Behavior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create sample data for testing
        cls.train_data = pd.DataFrame({
            'Date': pd.date_range(start='2021-01-01', periods=10, freq='D'),
            'StateHoliday': [0, 1, 0, 2, 0, 3, 0, 0, 0, 0],
            'Sales': [100, 200, 150, 250, 300, 350, 400, 450, 500, 550],
            'Customers': [10, 20, 15, 25, 30, 35, 40, 45, 50, 55],
            'DayOfWeek': [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            'Open': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })
        cls.test_data = pd.DataFrame({
            'Date': pd.date_range(start='2021-01-11', periods=10, freq='D'),
            'StateHoliday': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Sales': [150, 250, 200, 300, 350, 400, 450, 500, 550, 600],
            'Customers': [15, 25, 20, 30, 35, 40, 45, 50, 55, 60],
            'DayOfWeek': [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            'Open': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })

        # Save the sample data to CSV files
        cls.train_data.to_csv('test_train.csv', index=False)
        cls.test_data.to_csv('test_test.csv', index=False)

        # Initialize the EDA_Holiday_Seasonal_Behavior class with the sample data
        cls.eda_holiday = EDA_Holiday_Seasonal_Behavior('test_train.csv', 'test_test.csv')

    def test_analyze_sales_around_holidays(self):
        # Test the analyze_sales_around_holidays method
        try:
            holidays = self.eda_holiday.train[self.eda_holiday.train['StateHoliday'] != 0]['Date'].unique()
            avg_before, avg_during, avg_after = self.eda_holiday.analyze_sales_around_holidays(holidays)
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "analyze_sales_around_holidays method failed")

    def test_plot_sales_around_holidays(self):
        # Test the plot_sales_around_holidays method
        try:
            self.eda_holiday.plot_sales_around_holidays()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "plot_sales_around_holidays method failed")

    def test_plot_sales_around_specific_holidays(self):
        # Test the plot_sales_around_specific_holidays method
        try:
            self.eda_holiday.plot_sales_around_specific_holidays()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "plot_sales_around_specific_holidays method failed")

    def test_plot_customer_behavior(self):
        # Test the plot_customer_behavior method
        try:
            self.eda_holiday.plot_customer_behavior()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "plot_customer_behavior method failed")

    @classmethod
    def tearDownClass(cls):
        # Clean up the CSV files created for testing
        import os
        os.remove('test_train.csv')
        os.remove('test_test.csv')

if __name__ == '__main__':
    unittest.main()