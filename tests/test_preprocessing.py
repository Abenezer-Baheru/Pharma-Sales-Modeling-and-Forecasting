import unittest
import warnings
import pandas as pd
from scripts.preprocessing import DataProcessor

# Suppress warnings
warnings.filterwarnings("ignore")

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor('src/Data/train.csv', 'src/Data/test.csv', 'src/Data/store.csv')
        try:
            self.processor.load_data()
            self.processor.merge_data()
        except Exception as e:
            self.fail(f"Setup failed: {e}")

    def test_load_data(self):
        try:
            self.assertIsNotNone(self.processor.train)
            self.assertIsNotNone(self.processor.test)
            self.assertIsNotNone(self.processor.store)
            self.assertEqual(self.processor.train.shape[1], 18)  # Ensure all columns are loaded after merge
            self.assertEqual(self.processor.test.shape[1], 17)   # Ensure all columns are loaded after merge
            self.assertEqual(self.processor.store.shape[1], 10)  # Ensure all columns are loaded
        except Exception as e:
            self.fail(f"test_load_data failed: {e}")

    def test_merge_data(self):
        try:
            self.assertIn('StoreType', self.processor.train.columns)
            self.assertIn('StoreType', self.processor.test.columns)
            self.assertEqual(self.processor.train.shape[1], 18)  # Ensure columns are merged
            self.assertEqual(self.processor.test.shape[1], 17)   # Ensure columns are merged
        except Exception as e:
            self.fail(f"test_merge_data failed: {e}")

    def test_clean_data(self):
        try:
            self.processor.clean_data()
            self.assertFalse(self.processor.train.isnull().values.any())
            self.assertFalse(self.processor.test.isnull().values.any())
            self.assertTrue(self.processor.train['StateHoliday'].dtype == 'int64')
            self.assertTrue(self.processor.test['StateHoliday'].dtype == 'int64')
            self.assertTrue(self.processor.store['StoreType'].dtype == 'int64')
            self.assertTrue(self.processor.store['Assortment'].dtype == 'int64')
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.processor.train['Date']))
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.processor.test['Date']))
        except Exception as e:
            self.fail(f"test_clean_data failed: {e}")

    def test_visualize_missing_values(self):
        try:
            self.processor.visualize_missing_values()
        except Exception as e:
            self.fail(f"test_visualize_missing_values failed: {e}")

    def test_visualize_outliers(self):
        try:
            self.processor.visualize_outliers()
        except Exception as e:
            self.fail(f"test_visualize_outliers failed: {e}")

    def test_count_outliers(self):
        try:
            num_outliers = self.processor.count_outliers()
            self.assertIsInstance(num_outliers, int)
            print(f"Number of outliers: {num_outliers}")
        except Exception as e:
            self.fail(f"test_count_outliers failed: {e}")

if __name__ == '__main__':
    unittest.main()