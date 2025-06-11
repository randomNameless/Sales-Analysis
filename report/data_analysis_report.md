# Walmart Sales Analysis Report

**Author:** Bangyan Ju  
**Report Date:** June 10, 2025

---

## 1. Objective

This project analyzes Walmart's historical sales data to uncover trends, forecast future performance, and explore store-specific behaviors. Delivered as an interactive Streamlit web app.

---

## 2. Dataset Overview

- **File:** `Walmart_Sales.csv` from [Kaggle](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)
- **Records:** 6,435 weekly entries
- **Columns:**

| Column         | Type       | Description                           |
|----------------|------------|---------------------------------------|
| `Date`         | datetime   | Start of the week                     |
| `Weekly_Sales` | float      | Sales for that week (USD)            |
| `Store`        | int        | Store identifier (1–45)              |
| `Holiday_Flag` | int (0/1)  | Whether that week contains a holiday |
| `Temperature`  | float      | Average weekly temperature (°F)      |
| `Fuel_Price`   | float      | Average fuel price (USD)             |
| `CPI`          | float      | Consumer Price Index                 |
| `Unemployment` | float      | Unemployment rate (%)                |

---

## 3. Key Features Overview

### 3.1 Weekly Sales Over Time
- Aggregates sales data by week across all stores.
- Visualized using an interactive Plotly line chart.
- Reveals seasonal patterns and sales trends over time.

### 3.2 Holiday Impact on Sales
- Visualized the effect of holidays on sales using an interactive box plot.
- Enables quick comparison of sales distribution, highlighting variance during holiday periods.
- Shows how holidays affect weekly sales across all stores.

### 3.3 Time Series Forecasting (Prophet)
- Forecasts next 12 weeks using Facebook Prophet.
- Adjustable confidence intervals (80%, 90%, 95%).
- Visualized with Plotly, results exportable as CSV.
- Predicts future weekly sales based on past trends.

### 3.4 Anomaly Detection
- Uses a 4-week rolling average and standard deviation.
- Flags abnormally high sales periods.
- Visualized using combined line and scatter plot.
- Identifies unexpected spikes or drops in sales.

### 3.5 Holiday Sales Analysis
- Box plots comparing holiday vs non-holiday weeks.
- Offers statistical insight into holiday-driven sales behavior.

### 3.6 Monthly Heatmap by Store
- Pivot table heatmap: Store vs Month sales volume.
- Easily highlights seasonality and high-performing stores.
- Provides a bird's-eye view of store sales patterns month by month.

### 3.7 Store-Level Forecasting
- User can select any store for individual forecast.
- Prophet model is applied to selected store's data.
- Results are visualized and downloadable as CSV.
- Enables custom forecasting for a specific store's sales.

### 3.8 Store Clustering (KMeans)
- Clusters stores using `Total_Sales`, `Avg_Weekly_Sales`, and `Holiday_Impact`.
- Number of clusters (K) adjustable from 2 to 6.
- Features normalized with `StandardScaler`.
- Shows interactive scatter plot and optional cluster table.
- Groups similar stores to uncover operational or demographic similarities.

### 3.9 Feature Correlation Heatmap
- Computes correlation matrix among numeric variables.
- Visualized using Plotly heatmap with annotation.
- Highlights which features are strongly associated with sales.

### 3.10 Price Sensitivity Analysis (Regression)
- Interactive filters: CPI, Fuel Price, Holiday filter.
- Fits regression model: `Weekly_Sales ~ CPI + Fuel_Price`.
- Displays equation and R² score.
- Includes:
  - 2D scatter plots with trendlines.
  - 3D interactive surface plot (CPI × Fuel Price × Predicted Sales).
- Explores how price-related factors influence sales under different conditions.

### 3.11 Real-Time Sales Prediction Tool
- Accepts manual input for CPI and Fuel Price.
- Outputs predicted weekly sales instantly using trained model.
- Simulates expected sales under hypothetical market conditions.

---

## 4. Key Insights

- **Weekly Sales Over Time**  
  From February to November, weekly sales typically range between 40M and 50M.  
  There are noticeable small peaks around the 1st to 10th of each month, and dips around the 20th to 31st.  
  A medium-sized peak occurs in late November (around 60M–70M), and a significant spike appears between December 20–26, reaching 70M–80M.  
  These patterns indicate strong seasonality and consumer behavior cycles tied to monthly and holiday events.

- **Holiday Impact on Sales**  
  This box plot compares weekly sales distributions between holiday and non-holiday periods.  
  Median sales are visibly higher during holidays, showing that holidays generally drive stronger performance.  
  However, non-holiday periods display more high-value outliers, indicating occasional weeks with exceptional sales even outside official holiday seasons.  
  This suggests that while holidays are key drivers of revenue, targeted strategies during non-holiday weeks could also yield significant gains.

- **Prophet Forecast with Confidence Intervals**  
  This forecast visualizes weekly sales projections for the next 12 weeks using Facebook Prophet.  
  The blue line represents predicted sales, while the shaded area shows the confidence interval (e.g., 90%) reflecting uncertainty around the forecast.  
  The model captures seasonality and growth patterns based on historical trends.  
  A notable surge is expected around December 21, 2012 — aligning with past holiday peaks.

- **Anomaly Detection in Weekly Sales**  
  This chart identifies abnormal spikes in weekly sales using a 4-week rolling average and standard deviation.  
  Most anomalies appear from late November to late December, linked to Black Friday and Christmas demand surges.  
  This emphasizes the need for proactive resource planning during the holiday season.

- **Monthly Sales Heatmap per Store**  
  This heatmap highlights month-by-store sales variations.  
  Certain stores exhibit exceptionally high sales during specific months, likely due to regional factors or local events.  
  This suggests potential for deeper geographic or market-specific analysis.

- **Store Clustering (Unsupervised Learning)**  
  Stores were grouped based on total sales, average weekly sales, and holiday sales impact.  
  The results reveal clear segments of store behavior and performance.  
  Notably, stores **2, 4, 10, 13, 14, 20, and 27** show **high sales and strong holiday sensitivity**, making them ideal for **priority resource allocation** such as focused promotions and inventory planning.  
  Conversely, stores **3, 5, 33, 35, 38, and 44** show **low sales and low holiday sensitivity**, suggesting these locations may benefit from **targeted promotions or operational improvements**.  
  Such clustering enables more tailored strategies per segment.

- **Feature Correlation Heatmap**  
  Positive correlation between `Rolling_Mean` and `Weekly_Sales` suggests recent performance predicts current sales well.  
  Fuel price shows moderate correlation with year, consistent with long-term trends.  
  Negative correlation between `CPI` and `Weekly_Sales` implies inflation might suppress sales to some extent.

- **Price Sensitivity Analysis (Regression)**  
  Linear regression assessed the impact of CPI and fuel price on weekly sales.  
  The R² score was very low (~0.005), meaning these variables explain little sales variance.  
  Suggests price alone is not a strong driver — other operational or behavioral factors may be more important.

---

## 5. Business Recommendations

- **Prioritize holiday readiness**  
  Allocate more inventory, workforce, and logistics during high-sales periods, especially from late November to December.

- **Leverage store segmentation**  
  Customize promotions and operations per store cluster for better ROI and efficiency.

- **Track economic indicators cautiously**  
  While macroeconomic variables are informative, they are weak predictors of short-term sales. Use them in context.

- **Use forecast tools operationally**  
  Implement Prophet-based forecasts in planning inventory, staffing, and promotions 2–3 months ahead.

- **Investigate regional trends**  
  Deep dive into standout store-month combinations to uncover local advantages or missed opportunities.

---

## 6. Limitations

- **No regional granularity**  
  No geographic data is provided, so regional trends and differences are not directly analyzable.

- **Lacks product/category breakdown**  
  The dataset only includes aggregate store-level sales — product-level insights are unavailable.

- **Weak economic predictors**  
  The low R² score indicates CPI and fuel price are not sufficient to model weekly sales trends.

- **Historical scope is limited**  
  Data ends in 2012, which may affect model generalizability to current market behavior.

- **Customer behavior not included**  
  No demographic, loyalty, or behavioral data is available, which limits targeting and personalization insights.

---

## 7. Future Work

- **Enhance feature engineering**  
  Create additional variables such as rolling averages, holiday countdowns, and month/week encodings to better capture sales drivers.

- **Expand the dataset**  
  Include more recent data beyond 2012 and enrich it with geographic, product-level, and promotional features for deeper analysis.

- **Test advanced models**  
  Apply non-linear or ensemble models such as Random Forest, XGBoost, or LightGBM to improve predictive accuracy over linear baselines.

- **Implement regional analysis**  
  If location information becomes available, analyze store performance across regions to uncover localized trends and opportunities.

- **Customer segmentation**  
  Introduce behavioral or demographic data (if obtainable) to enable personalized marketing strategies and better demand targeting.

- **Validate with real promotions**  
  Backtest model performance and anomaly detection against known promotional periods to evaluate business impact.

- **Build real-time dashboard**  
  Connect the app to a live database or API to support continuous updates, automated monitoring, and business decision-making in real time.

---

## 8. Links

- [Live Streamlit App](https://sales-analysis-bangyanju.streamlit.app)
- [GitHub Repository](https://github.com/randomNameless/Sales-Analysis)

---

> End of Report

