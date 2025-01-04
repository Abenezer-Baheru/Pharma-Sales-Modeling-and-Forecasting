"""
This script will cover:

- Check for distribution in both training and test sets - are the promotions distributed similarly between these two groups?
- What can you say about the correlation between sales and the number of customers?
- How does promo affect sales? Are the promos attracting more customers? How does it affect already existing customers?
- Could the promos be deployed in more effective ways? Which stores should promos be deployed in? 

"""
# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tabulate import tabulate

# Configure logging to display messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EDA_Distribution_Promotions:
    def __init__(self, train_path, test_path):
        try:
            self.train = pd.read_csv(train_path)
            self.test = pd.read_csv(test_path)
            self.train['Date'] = pd.to_datetime(self.train['Date'])
            self.test['Date'] = pd.to_datetime(self.test['Date'])
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")

    def plot_promo_distribution(self):
        try:
            # Calculate the distribution of promotions in the training set
            train_promo_dist = self.train['Promo'].value_counts()
            # Calculate the distribution of promotions in the test set
            test_promo_dist = self.test['Promo'].value_counts()

            # Display the distribution of promotions in tabular format
            train_promo_table = tabulate(train_promo_dist.reset_index(), headers=['Promo', 'Count'], tablefmt='grid')
            test_promo_table = tabulate(test_promo_dist.reset_index(), headers=['Promo', 'Count'], tablefmt='grid')
            logging.info("Distribution of Promotions in Training Set:\n" + train_promo_table)
            logging.info("Distribution of Promotions in Test Set:\n" + test_promo_table)

            # Create a single figure with two subplots for the donut plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Plot the distribution of promotions in the training set
            ax1.pie(train_promo_dist, labels=train_promo_dist.index, autopct='%1.1f%%', startangle=140, colors=['blue', 'lightblue'])
            centre_circle1 = plt.Circle((0, 0), 0.70, fc='white')
            ax1.add_artist(centre_circle1)
            ax1.set_title('Distribution of Promotions in Training Set')

            # Plot the distribution of promotions in the test set
            ax2.pie(test_promo_dist, labels=test_promo_dist.index, autopct='%1.1f%%', startangle=140, colors=['red', 'lightcoral'])
            centre_circle2 = plt.Circle((0, 0), 0.70, fc='white')
            ax2.add_artist(centre_circle2)
            ax2.set_title('Distribution of Promotions in Test Set')

            # Display the plots
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting promotion distribution: {e}")

    def plot_sales_vs_customers(self):
        try:
            correlation = self.train['Sales'].corr(self.train['Customers'])
            logging.info(f"Correlation coefficient between Sales and Customers: {correlation:.2f}")

            plt.figure(figsize=(10, 6))
            sns.regplot(x='Customers', y='Sales', data=self.train, scatter_kws={'s': 10}, line_kws={'color': 'red'})
            plt.title('Scatter Plot of Sales vs. Number of Customers')
            plt.xlabel('Number of Customers')
            plt.ylabel('Sales')
            plt.grid(True)
            plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting sales vs customers: {e}")

    def plot_promo_effects(self):
        try:
            avg_sales_with_promo = self.train[self.train['Promo'] == 1]['Sales'].mean()
            avg_sales_without_promo = self.train[self.train['Promo'] == 0]['Sales'].mean()
            avg_customers_with_promo = self.train[self.train['Promo'] == 1]['Customers'].mean()
            avg_customers_without_promo = self.train[self.train['Promo'] == 0]['Customers'].mean()
            avg_sales_per_customer_with_promo = self.train[self.train['Promo'] == 1]['Sales'].sum() / self.train[self.train['Promo'] == 1]['Customers'].sum()
            avg_sales_per_customer_without_promo = self.train[self.train['Promo'] == 0]['Sales'].sum() / self.train[self.train['Promo'] == 0]['Customers'].sum()

            promo_effect_table = tabulate([
                ['Average Sales with Promotion', avg_sales_with_promo],
                ['Average Sales without Promotion', avg_sales_without_promo],
                ['Average Number of Customers with Promotion', avg_customers_with_promo],
                ['Average Number of Customers without Promotion', avg_customers_without_promo],
                ['Average Sales per Customer with Promotion', avg_sales_per_customer_with_promo],
                ['Average Sales per Customer without Promotion', avg_sales_per_customer_without_promo]
            ], headers=['Metric', 'Value'], tablefmt='grid')
            logging.info("Promotion Effects:\n" + promo_effect_table)

            promo_effect_df = pd.DataFrame({
                'Metric': ['Sales', 'Sales', 'Customers', 'Customers', 'Sales per Customer', 'Sales per Customer'],
                'Promotion': ['With Promo', 'Without Promo', 'With Promo', 'Without Promo', 'With Promo', 'Without Promo'],
                'Value': [avg_sales_with_promo, avg_sales_without_promo, avg_customers_with_promo, avg_customers_without_promo, avg_sales_per_customer_with_promo, avg_sales_per_customer_without_promo]
            })

            plt.figure(figsize=(14, 8))
            sns.barplot(data=promo_effect_df, x='Metric', y='Value', hue='Promotion', palette='viridis')
            plt.title('Effect of Promotions on Sales, Customer Count, and Sales per Customer')
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.legend(title='Promotion')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting promotion effects: {e}")

    def plot_store_promo_effects(self):
        try:
            store_promo_effect = self.train.groupby(['Store', 'Promo']).agg({
                'Sales': 'mean',
                'Customers': 'mean'
            }).reset_index()

            store_promo_effect_pivot = store_promo_effect.pivot(index='Store', columns='Promo', values=['Sales', 'Customers'])
            store_promo_effect_pivot.columns = ['Sales_Without_Promo', 'Sales_With_Promo', 'Customers_Without_Promo', 'Customers_With_Promo']
            store_promo_effect_pivot = store_promo_effect_pivot.reset_index()

            store_promo_effect_pivot['Sales_Impact'] = store_promo_effect_pivot['Sales_With_Promo'] - store_promo_effect_pivot['Sales_Without_Promo']
            store_promo_effect_pivot['Customers_Impact'] = store_promo_effect_pivot['Customers_With_Promo'] - store_promo_effect_pivot['Customers_Without_Promo']

            store_promo_effect_table = tabulate(store_promo_effect_pivot.head(), headers='keys', tablefmt='grid')
            logging.info("Store Promotion Effects:\n" + store_promo_effect_table)

            top_stores_by_sales_impact = store_promo_effect_pivot.sort_values(by='Sales_Impact', ascending=False).head(10)
            top_stores_by_customers_impact = store_promo_effect_pivot.sort_values(by='Customers_Impact', ascending=False).head(10)

            top_sales_impact_table = tabulate(top_stores_by_sales_impact[['Store', 'Sales_Impact']], headers='keys', tablefmt='grid')
            top_customers_impact_table = tabulate(top_stores_by_customers_impact[['Store', 'Customers_Impact']], headers='keys', tablefmt='grid')
            logging.info("Top 10 Stores by Sales Impact:\n" + top_sales_impact_table)
            logging.info("Top 10 Stores by Customers Impact:\n" + top_customers_impact_table)

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            sns.barplot(data=top_stores_by_sales_impact, x='Store', y='Sales_Impact', palette='viridis', ax=axes[0])
            axes[0].set_title('Top 10 Stores by Sales Impact from Promotions', fontsize=20)
            axes[0].set_xlabel('Store', fontsize=16)
            axes[0].set_ylabel('Sales Impact', fontsize=16)
            axes[0].tick_params(axis='x', rotation=45, labelsize=14)
            axes[0].tick_params(axis='y', labelsize=14)
            axes[0].grid(True)

            sns.barplot(data=top_stores_by_customers_impact, x='Store', y='Customers_Impact', palette='viridis', ax=axes[1])
            axes[1].set_title('Top 10 Stores by Customers Impact from Promotions', fontsize=20)
            axes[1].set_xlabel('Store', fontsize=16)
            axes[1].set_ylabel('Customers Impact', fontsize=16)
            axes[1].tick_params(axis='x', rotation=45, labelsize=14)
            axes[1].tick_params(axis='y', labelsize=14)
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting store promotion effects: {e}")

# Example usage:
# eda = EDA_Distribution_Promotions('../src/Data/cleaned_train.csv', '../src/Data/cleaned_test.csv')
# eda.plot_promo_distribution()
# eda.plot_sales_vs_customers()
# eda.plot_promo_effects()
# eda.plot_store_promo_effects()
