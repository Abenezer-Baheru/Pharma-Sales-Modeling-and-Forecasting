# Pharma-Sales-Modeling-and-Forecasting
### Exploratory Data Analysis (EDA) Project

This project performs exploratory data analysis (EDA) on retail store sales data. The analysis is divided into three main modules:

1. **EDA_Distribution_Promotions.py**: Analyzes the distribution of promotions and their effects on sales and customer behavior.
2. **EDA_Holiday_Seasonal_Behavior.py**: Analyzes sales behavior before, during, and after holidays, as well as seasonal purchase behaviors.
3. **EDA_Store_Competitor_Analysis.py**: Analyzes the impact of store competitors, assortment types, and store opening times on sales.

### Project Structure

 ├── EDA_Distribution_Promotions.py ├── EDA_Holiday_Seasonal_Behavior.py ├── EDA_Store_Competitor_Analysis.py ├── EDA.ipynb ├── test_EDA_Distribution_Promotions.py ├── test_EDA_Holiday_Seasonal_Behavior.py ├── test_EDA_Store_Competitor_Analysis.py ├── README.md └── data ├── cleaned_train.csv └── cleaned_test.csv

 
## Setup Instructions

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-repo/eda-project.git
    cd eda-project
    ```

2. **Install the required packages**:
    ```sh
    pip install pandas matplotlib seaborn tabulate
    ```

3. **Ensure the data files are in the `data` directory**:
    - `cleaned_train.csv`
    - `cleaned_test.csv`

## Usage

### Running the EDA Scripts

1. **EDA_Distribution_Promotions.py**:
    ```python
    from EDA_Distribution_Promotions import EDA_Distribution_Promotions

    eda_promo = EDA_Distribution_Promotions('data/cleaned_train.csv', 'data/cleaned_test.csv')
    eda_promo.plot_promo_distribution()
    eda_promo.plot_sales_vs_customers()
    eda_promo.plot_promo_effects()
    eda_promo.plot_store_promo_effects()
    ```

2. **EDA_Holiday_Seasonal_Behavior.py**:
    ```python
    from EDA_Holiday_Seasonal_Behavior import EDA_Holiday_Seasonal_Behavior

    eda_holiday = EDA_Holiday_Seasonal_Behavior('data/cleaned_train.csv', 'data/cleaned_test.csv')
    eda_holiday.plot_sales_around_holidays()
    eda_holiday.plot_sales_around_specific_holidays()
    eda_holiday.plot_customer_behavior()
    ```

3. **EDA_Store_Competitor_Analysis.py**:
    ```python
    from EDA_Store_Competitor_Analysis import EDA_Store_Competitor_Analysis

    eda_store_competitor = EDA_Store_Competitor_Analysis('data/cleaned_train.csv')
    eda_store_competitor.analyze_weekday_weekend_sales()
    eda_store_competitor.analyze_sales_diff_weekday_weekend()
    eda_store_competitor.analyze_assortment_sales()
    eda_store_competitor.analyze_competitor_distance_sales()
    eda_store_competitor.analyze_competitor_opening_effect()
    eda_store_competitor.analyze_competitor_opening_effect_all()
    ```

### Running the Jupyter Notebook

1. **EDA.ipynb**:
    - Open the notebook in Jupyter:
        ```sh
        jupyter notebook EDA.ipynb
        ```
    - Follow the instructions and run the cells to perform the analysis.

### Running the Unit Tests

1. **test_EDA_Distribution_Promotions.py**:
    ```sh
    python test_EDA_Distribution_Promotions.py
    ```

2. **test_EDA_Holiday_Seasonal_Behavior.py**:
    ```sh
    python test_EDA_Holiday_Seasonal_Behavior.py
    ```

3. **test_EDA_Store_Competitor_Analysis.py**:
    ```sh
    python test_EDA_Store_Competitor_Analysis.py
    ```

## Description of Analyses

### EDA_Distribution_Promotions.py

- **plot_promo_distribution**: Checks for distribution in both training and test sets to see if the promotions are distributed similarly between these two groups.
- **plot_sales_vs_customers**: Analyzes the correlation between sales and the number of customers.
- **plot_promo_effects**: Analyzes how promotions affect sales, whether they attract more customers, and how they affect existing customers.
- **plot_store_promo_effects**: Analyzes if promotions could be deployed in more effective ways and identifies which stores should have promotions deployed.

### EDA_Holiday_Seasonal_Behavior.py

- **plot_sales_around_holidays**: Checks and compares sales behavior before, during, and after holidays.
- **plot_sales_around_specific_holidays**: Analyzes purchase behaviors during specific seasonal holidays such as Christmas and Easter.
- **plot_customer_behavior**: Analyzes trends of customer behavior during store opening and closing times.

### EDA_Store_Competitor_Analysis.py

- **analyze_weekday_weekend_sales**: Identifies stores open on all weekdays and analyzes their weekend sales.
- **analyze_sales_diff_weekday_weekend**: Analyzes the sales difference between weekday and weekend for stores open on all weekdays.
- **analyze_assortment_sales**: Analyzes how the assortment type affects sales.
- **analyze_competitor_distance_sales**: Analyzes how the distance to the next competitor affects sales and whether the distance matters if the store and its competitors are in city centers.
- **analyze_competitor_opening_effect**: Analyzes how the opening or reopening of new competitors affects stores, especially those with initially NA competitor distance but later have values for competitor distance.
- **analyze_competitor_opening_effect_all**: Analyzes the effect of competitor openings on sales for all stores with competitor distance information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.