
  # 🧾 Walmart Sales Forecasting & Analysis Dashboard

  An end-to-end interactive data science dashboard built with **Streamlit**, focused on Walmart's historical sales data. This project was created to demonstrate practical data science and machine learning skills for solving real-world business problems — including time series forecasting, regression modeling, anomaly detection, and clustering analysis.

  > 🔗 **Live App**: [Click to Launch](https://sales-analysis-bangyanju.streamlit.app/)

  ---

  ## 📊 What This Dashboard Does

  This Streamlit app enables users to:
  - 📅 Visualize weekly sales trends over time
  - 🎉 Explore the impact of holidays on sales
  - 📈 Forecast future sales using **Facebook Prophet**
  - 📉 Analyze price sensitivity to CPI and fuel price
  - 🔍 Detect anomalies in historical sales data
  - 🏪 Cluster stores based on their sales profiles
  - 🔮 Predict sales using a custom regression tool

  ---

## 📁 Project Structure

```
Sales-Analysis/
├── sales_analysis_app.py        # Main Streamlit app
├── requirements.txt             # Python dependencies
├── data/
│   └── Walmart_Sales.csv        # Sample data
├── report/
│   └── data_analysis_report.pdf # Formal report
├── README.md                    # This file
```


## 🚀 Key Features

| Feature                          | Description |
|----------------------------------|-------------|
| **Prophet Forecast**             | Predict 12-week sales with confidence intervals |
| **Anomaly Detection**            | Spot unusual sales spikes using rolling averages |
| **Holiday Impact Analysis**      | Compare sales during holiday vs non-holiday weeks |
| **Store-Level Forecasting**      | Select any store and view its individual forecast |
| **Regression + 3D Visualization**| CPI & Fuel Price regression with 3D surface |
| **KMeans Store Clustering**      | Unsupervised learning to segment stores |
| **Downloadable Reports**         | Forecast and cluster outputs as CSV |

---

## 📦 How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Sales-Analysis.git
   cd Sales-Analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 📄 Sample Data


  ### 📄 Sample Data Format

  If you don’t upload your own file, the app uses a default dataset:  
  `data/Walmart_Sales.csv`

  To test custom behavior, you can upload your **own CSV** with the **following columns and matching data types**:

  | Column          | Data Type       | Description                              |
  |-----------------|------------------|------------------------------------------|
  | `Date`          | `datetime`       | Week start date (e.g., `2010-02-05`)     |
  | `Weekly_Sales`  | `float`          | Sales amount for that week               |
  | `Store`         | `int`            | Store ID (e.g., `1`, `2`, ..., `45`)     |
  | `Holiday_Flag`  | `int` (0 or 1)   | Indicates if the week includes a holiday |
  | `Temperature`   | `float`          | Average temperature that week            |
  | `Fuel_Price`    | `float`          | Price of fuel that week                  |
  | `CPI`           | `float`          | Consumer Price Index                     |
  | `Unemployment`  | `float`          | Unemployment rate                        |

  **Note:**
  - `Date` must be parsable by `pandas.to_datetime()`.
  - `Holiday_Flag` must be binary (`0` = Non-Holiday, `1` = Holiday).
  - All columns must be present with appropriate data types.


---

## 📑 Report

A formal **data analysis report** is available in the `report/` folder. It outlines:
- Business context
- Analytical steps
- Key insights & visualizations
- Forecasting accuracy

---

## 📬 Author

**Bangyan Ju**  
PhD Student, University of Cincinnati 
📧 [jubn@mai.uc.edu]  

---


## 📝 License

This project is licensed under the MIT License — feel free to reuse or extend with credit.
