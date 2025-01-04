import unittest
import pandas as pd
from EDA_Store_Competitor_Analysis import EDA_Store_Competitor_Analysis

class TestEDA_Store_Competitor_Analysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create sample data for testing
        cls.train_data = pd.DataFrame({
            'Date': pd.date_range(start='2021-01-01', periods=10, freq='D'),
            'DayOfWeek': [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            'Sales': [100, 200, 150, 250, 300, 350, 400, 450, 500, 550],
            'Customers': [10, 20, 15, 25, 30, 35, 40, 45, 50, 55],
            'Open': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'Store': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'Assortment': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
            'CompetitionDistance': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'StoreType': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
        })

        # Save the sample data to a CSV file
        cls.train_data.to_csv('test_train.csv', index=False)

        # Initialize the EDA_Store_Competitor_Analysis class with the sample data
        cls.eda_store_competitor = EDA_Store_Competitor_Analysis('test_train.csv')

    def test_analyze_weekday_weekend_sales(self):
        # Test the analyze_weekday_weekend_sales method
        try:
            self.eda_store_competitor.analyze_weekday_weekend_sales()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "analyze_weekday_weekend_sales method failed")

    def test_analyze_sales_diff_weekday_weekend(self):
        # Test the analyze_sales_diff_weekday_weekend method
        try:
            self.eda_store_competitor.analyze_sales_diff_weekday_weekend()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "analyze_sales_diff_weekday_weekend method failed")

    def test_analyze_assortment_sales(self):
        # Test the analyze_assortment_sales method
        try:
            self.eda_store_competitor.analyze_assortment_sales()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "analyze_assortment_sales method failed")

    def test_analyze_competitor_distance_sales(self):
        # Test the analyze_competitor_distance_sales method
        try:
            self.eda_store_competitor.analyze_competitor_distance_sales()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "analyze_competitor_distance_sales method failed")

    def test_analyze_competitor_opening_effect(self):
        # Test the analyze_competitor_opening_effect method
        try:
            self.eda_store_competitor.analyze_competitor_opening_effect()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "analyze_competitor_opening_effect method failed")

    def test_analyze_competitor_opening_effect_all(self):
        # Test the analyze_competitor_opening_effect_all method
        try:
            self.eda_store_competitor.analyze_competitor_opening_effect_all()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "analyze_competitor_opening_effect_all method failed")

    @classmethod
    def tearDownClass(cls):
        # Clean up the CSV file created for testing
        import os
        os.remove('test_train.csv')

if __name__ == '__main__':
    unittest.main()