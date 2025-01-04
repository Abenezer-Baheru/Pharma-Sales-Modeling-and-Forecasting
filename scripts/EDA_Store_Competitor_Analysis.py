# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tabulate import tabulate

# Configure logging to display messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EDA_Store_Competitor_Analysis:
    def __init__(self, train_path):
        try:
            self.train = pd.read_csv(train_path)
            self.train['Date'] = pd.to_datetime(self.train['Date'])
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")

    def analyze_weekday_weekend_sales(self):
        try:
            weekdays = [0, 1, 2, 3, 4]  # Monday to Friday
            stores_open_all_weekdays = self.train[self.train['DayOfWeek'].isin(weekdays) & (self.train['Open'] == 1)].groupby('Store')['DayOfWeek'].nunique()
            stores_open_all_weekdays = stores_open_all_weekdays[stores_open_all_weekdays == 5].index.tolist()

            logging.info(f"Stores open on all weekdays: {stores_open_all_weekdays}")

            weekends = [5, 6]  # Saturday and Sunday
            weekend_sales = self.train[(self.train['Store'].isin(stores_open_all_weekdays)) & (self.train['DayOfWeek'].isin(weekends))]
            weekend_sales_avg = weekend_sales.groupby(['Store', 'DayOfWeek'])['Sales'].mean().reset_index()
            weekend_sales_avg['DayOfWeek'] = weekend_sales_avg['DayOfWeek'].map({5: 'Saturday', 6: 'Sunday'})

            logging.info(f"Weekend sales average data:\n{weekend_sales_avg}")

            weekend_sales_table = tabulate(weekend_sales_avg, headers='keys', tablefmt='grid')
            logging.info("Weekend Sales Average:\n" + weekend_sales_table)

            plt.figure(figsize=(14, 8))
            sns.barplot(data=weekend_sales_avg, x='DayOfWeek', y='Sales', hue='Store', palette='viridis')
            plt.title('Average Sales on Weekends for Stores Open on All Weekdays', fontsize=20, fontweight='bold')
            plt.xlabel('Day of the Week', fontsize=16, fontweight='bold')
            plt.ylabel('Average Sales', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45)
            plt.legend(title='Store', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Error analyzing weekday and weekend sales: {e}")

    def analyze_sales_diff_weekday_weekend(self):
        try:
            weekdays = [0, 1, 2, 3, 4]  # Monday to Friday
            stores_open_all_weekdays = self.train[self.train['DayOfWeek'].isin(weekdays) & (self.train['Open'] == 1)].groupby('Store')['DayOfWeek'].nunique()
            stores_open_all_weekdays = stores_open_all_weekdays[stores_open_all_weekdays == 5].index.tolist()

            weekday_sales = self.train[self.train['DayOfWeek'].isin(weekdays) & (self.train['Store'].isin(stores_open_all_weekdays))]
            avg_weekday_sales = weekday_sales.groupby('Store')['Sales'].mean().reset_index()
            avg_weekday_sales.rename(columns={'Sales': 'Avg_Weekday_Sales'}, inplace=True)

            weekends = [5, 6]  # Saturday and Sunday
            weekend_sales = self.train[self.train['DayOfWeek'].isin(weekends) & (self.train['Store'].isin(stores_open_all_weekdays))]
            avg_weekend_sales = weekend_sales.groupby('Store')['Sales'].mean().reset_index()
            avg_weekend_sales.rename(columns={'Sales': 'Avg_Weekend_Sales'}, inplace=True)

            sales_diff = pd.merge(avg_weekday_sales, avg_weekend_sales, on='Store')
            sales_diff['Sales_Diff'] = sales_diff['Avg_Weekday_Sales'] - sales_diff['Avg_Weekend_Sales']

            sales_diff_table = tabulate(sales_diff.head(), headers='keys', tablefmt='grid')
            logging.info("Sales Difference (Weekday vs Weekend):\n" + sales_diff_table)

            top_10_affected_stores = sales_diff.nlargest(10, 'Sales_Diff')
            bottom_10_affected_stores = sales_diff.nsmallest(10, 'Sales_Diff')

            top_10_affected_table = tabulate(top_10_affected_stores, headers='keys', tablefmt='grid')
            bottom_10_affected_table = tabulate(bottom_10_affected_stores, headers='keys', tablefmt='grid')
            logging.info("Top 10 Stores Most Affected:\n" + top_10_affected_table)
            logging.info("Bottom 10 Stores Least Affected:\n" + bottom_10_affected_table)

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            sns.barplot(data=top_10_affected_stores.melt(id_vars='Store', value_vars=['Avg_Weekday_Sales', 'Avg_Weekend_Sales']), x='Store', y='value', hue='variable', palette='viridis', ax=axes[0])
            axes[0].set_title('Average Sales for Top 10 Stores Most Affected', fontsize=20, fontweight='bold')
            axes[0].set_xlabel('Store', fontsize=16, fontweight='bold')
            axes[0].set_ylabel('Average Sales', fontsize=16, fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].legend(title='Sales Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True)

            sns.barplot(data=bottom_10_affected_stores.melt(id_vars='Store', value_vars=['Avg_Weekday_Sales', 'Avg_Weekend_Sales']), x='Store', y='value', hue='variable', palette='viridis', ax=axes[1])
            axes[1].set_title('Average Sales for Bottom 10 Stores Least Affected', fontsize=20, fontweight='bold')
            axes[1].set_xlabel('Store', fontsize=16, fontweight='bold')
            axes[1].set_ylabel('Average Sales', fontsize=16, fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend(title='Sales Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error analyzing sales difference between weekday and weekend: {e}")

    def analyze_assortment_sales(self):
        try:
            assortment_sales = self.train.groupby('Assortment')['Sales'].mean().reset_index()
            assortment_sales_table = tabulate(assortment_sales, headers='keys', tablefmt='grid')
            logging.info("Assortment Sales:\n" + assortment_sales_table)

            plt.figure(figsize=(14, 8))
            sns.barplot(data=assortment_sales, x='Assortment', y='Sales', palette='viridis')
            plt.title('Average Sales by Assortment Type', fontsize=20, fontweight='bold')
            plt.xlabel('Assortment Type', fontsize=16, fontweight='bold')
            plt.ylabel('Average Sales', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Error analyzing assortment sales: {e}")

    def analyze_competitor_distance_sales(self):
        try:
            distance_sales = self.train.groupby('CompetitionDistance')['Sales'].mean().reset_index()
            distance_sales_table = tabulate(distance_sales, headers='keys', tablefmt='grid')
            logging.info("Distance Sales:\n" + distance_sales_table)

            city_center_stores = self.train[self.train['StoreType'] == 'c']  # Assuming 'c' represents city center stores
            city_center_distance_sales = city_center_stores.groupby('CompetitionDistance')['Sales'].mean().reset_index()
            city_center_distance_sales_table = tabulate(city_center_distance_sales, headers='keys', tablefmt='grid')
            logging.info("City Center Distance Sales:\n" + city_center_distance_sales_table)

            plt.figure(figsize=(14, 8))
            sns.lineplot(data=distance_sales, x='CompetitionDistance', y='Sales', marker='o', label='All Stores')
            plt.title('Average Sales by Distance to Next Competitor', fontsize=20, fontweight='bold')
            plt.xlabel('Distance to Next Competitor (meters)', fontsize=16, fontweight='bold')
            plt.ylabel('Average Sales', fontsize=16, fontweight='bold')
            plt.xticks(fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.grid(True)
            plt.legend()
            plt.show()

            plt.figure(figsize=(14, 8))
            sns.lineplot(data=city_center_distance_sales, x='CompetitionDistance', y='Sales', marker='o', label='City Center Stores')
            plt.title('Average Sales by Distance to Next Competitor (City Center Stores)', fontsize=20, fontweight='bold')
            plt.xlabel('Distance to Next Competitor (meters)', fontsize=16, fontweight='bold')
            plt.ylabel('Average Sales', fontsize=16, fontweight='bold')
            plt.xticks(fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.grid(True)
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f"Error analyzing competitor distance sales: {e}")

        try:
            distance_sales_by_type = self.train.groupby(['StoreType', 'CompetitionDistance'])['Sales'].mean().reset_index()
            logging.info(f"Distance sales by store type data:\n{distance_sales_by_type}")

            plt.figure(figsize=(14, 8))
            sns.lineplot(data=distance_sales_by_type, x='CompetitionDistance', y='Sales', hue='StoreType', marker='o')
            plt.title('Average Sales by Distance to Next Competitor for Each Store Type', fontsize=20, fontweight='bold')
            plt.xlabel('Distance to Next Competitor (meters)', fontsize=16, fontweight='bold')
            plt.ylabel('Average Sales', fontsize=16, fontweight='bold')
            plt.xticks(fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.grid(True)
            plt.legend(title='Store Type')
            plt.show()
        except Exception as e:
            logging.error(f"Error analyzing competitor distance sales by store type: {e}")

    def analyze_competitor_opening_effect(self):
        try:
            stores_with_na_competitor = self.train[self.train['CompetitionDistance'].isna()]['Store'].unique()
            stores_with_competitor_later = self.train[self.train['Store'].isin(stores_with_na_competitor) & self.train['CompetitionDistance'].notna()]['Store'].unique()
            affected_stores = self.train[self.train['Store'].isin(stores_with_competitor_later)]

            logging.info(f"Stores initially with NA competitor distance but later have values: {stores_with_competitor_later}")
            logging.info(affected_stores.head())

            affected_stores['CompetitorOpened'] = affected_stores['CompetitionDistance'].notna()
            sales_before_after = affected_stores.groupby(['Store', 'CompetitorOpened'])['Sales'].mean().reset_index()
            sales_before_after_table = tabulate(sales_before_after, headers='keys', tablefmt='grid')
            logging.info("Sales Before and After Competitor Opening:\n" + sales_before_after_table)

            plt.figure(figsize=(14, 8))
            sns.barplot(data=sales_before_after, x='Store', y='Sales', hue='CompetitorOpened', palette='viridis')
            plt.title('Average Sales Before and After Competitor Opening', fontsize=20, fontweight='bold')
            plt.xlabel('Store', fontsize=16, fontweight='bold')
            plt.ylabel('Average Sales', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.legend(title='Competitor Opened', labels=['Before', 'After'])
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Error analyzing competitor opening effect: {e}")

    def analyze_competitor_opening_effect_all(self):
        try:
            stores_with_competitor_info = self.train[self.train['CompetitionDistance'].notna()]
            logging.info(f"Number of records with competitor distance information: {len(stores_with_competitor_info)}")
            logging.info(stores_with_competitor_info.head())

            stores_with_competitor_info['CompetitionOpenDate'] = pd.to_datetime(
                stores_with_competitor_info['CompetitionOpenSinceYear'].astype(str) + '-' +
                stores_with_competitor_info['CompetitionOpenSinceMonth'].astype(str) + '-01',
                errors='coerce'
            )

            stores_with_competitor_info['CompetitorOpened'] = stores_with_competitor_info['Date'] >= stores_with_competitor_info['CompetitionOpenDate']
            sales_before_after = stores_with_competitor_info.groupby(['Store', 'CompetitorOpened'])['Sales'].mean().reset_index()
            sales_before_after_table = tabulate(sales_before_after, headers='keys', tablefmt='grid')
            logging.info("Sales Before and After Competitor Opening (All Stores):\n" + sales_before_after_table)

            plt.figure(figsize=(14, 8))
            sns.barplot(data=sales_before_after, x='Store', y='Sales', hue='CompetitorOpened', palette='viridis')
            plt.title('Average Sales Before and After Competitor Opening', fontsize=20, fontweight='bold')
            plt.xlabel('Store', fontsize=16, fontweight='bold')
            plt.ylabel('Average Sales', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, fontsize=14, fontweight='bold')
            plt.yticks(fontsize=14, fontweight='bold')
            plt.legend(title='Competitor Opened', labels=['Before', 'After'])
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Error analyzing competitor opening effect for all stores: {e}")

# Example usage:
# eda_store_competitor = EDA_Store_Competitor_Analysis('../src/Data/cleaned_train.csv')
# eda_store_competitor.analyze_weekday_weekend_sales()
# eda_store_competitor.analyze_sales_diff_weekday_weekend()
# eda_store_competitor.analyze_assortment_sales()
# eda_store_competitor.analyze_competitor_distance_sales()
# eda_store_competitor.analyze_competitor_opening_effect()
# eda_store_competitor.analyze_competitor_opening_effect_all()