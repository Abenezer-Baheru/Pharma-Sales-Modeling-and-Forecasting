import unittest
import pandas as pd
from EDA_Distribution_Promotions import EDA_Distribution_Promotions

class TestEDA_Distribution_Promotions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create sample data for testing
        cls.train_data = pd.DataFrame({
            'Date': pd.date_range(start='2021-01-01', periods=10, freq='D'),
            'Promo': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'Sales': [100, 200, 150, 250, 300, 350, 400, 450, 500, 550],
            'Customers': [10, 20, 15, 25, 30, 35, 40, 45, 50, 55],
            'Store': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })
        cls.test_data = pd.DataFrame({
            'Date': pd.date_range(start='2021-01-11', periods=10, freq='D'),
            'Promo': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'Sales': [150, 250, 200, 300, 350, 400, 450, 500, 550, 600],
            'Customers': [15, 25, 20, 30, 35, 40, 45, 50, 55, 60],
            'Store': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        })

        # Save the sample data to CSV files
        cls.train_data.to_csv('test_train.csv', index=False)
        cls.test_data.to_csv('test_test.csv', index=False)

        # Initialize the EDA_Distribution_Promotions class with the sample data
        cls.eda_promo = EDA_Distribution_Promotions('test_train.csv', 'test_test.csv')

    def test_plot_promo_distribution(self):
        # Test the plot_promo_distribution method
        try:
            self.eda_promo.plot_promo_distribution()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "plot_promo_distribution method failed")

    def test_plot_sales_vs_customers(self):
        # Test the plot_sales_vs_customers method
        try:
            self.eda_promo.plot_sales_vs_customers()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "plot_sales_vs_customers method failed")

    def test_plot_promo_effects(self):
        # Test the plot_promo_effects method
        try:
            self.eda_promo.plot_promo_effects()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "plot_promo_effects method failed")

    def test_plot_store_promo_effects(self):
        # Test the plot_store_promo_effects method
        try:
            self.eda_promo.plot_store_promo_effects()
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result, "plot_store_promo_effects method failed")

    @classmethod
    def tearDownClass(cls):
        # Clean up the CSV files created for testing
        import os
        os.remove('test_train.csv')
        os.remove('test_test.csv')

if __name__ == '__main__':
    unittest.main()