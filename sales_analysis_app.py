import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import plotly.graph_objects as go
import os
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


st.set_page_config(page_title="Walmart Sales Analysis", layout="wide")
st.title("Walmart Sales Forecasting & Analysis")

# --- Upload Data ---
st.sidebar.header("Upload Sales Analysis File")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
if uploaded_file is not None:
    sales_df = pd.read_csv(uploaded_file)
    filename = uploaded_file.name
else:
    DEFAULT_FILE = "data/Walmart_Sales.csv"  # <-- Make sure this file exists
    sales_df = pd.read_csv(DEFAULT_FILE)
    filename = os.path.basename(DEFAULT_FILE)
# Parse date column
sales_df['Date'] = pd.to_datetime(sales_df['Date'], dayfirst=True)
# Show which file is in use
st.info(f"Analyzing: **{filename}**")
# Side bar options
show_raw = st.sidebar.checkbox("Show Raw Data")
show_info = st.sidebar.checkbox("Show Data Info")
show_desc = st.sidebar.checkbox("Show Description")
if show_raw:
        st.subheader("Raw Data")
        st.dataframe(sales_df)
if show_info:
    st.subheader("Data Info")
    info_df = pd.DataFrame({
        'Column': sales_df.columns,
        'Non-Null Count': sales_df.notnull().sum().values,
        'Dtype': sales_df.dtypes.values.astype(str)
    })
    st.dataframe(info_df)
if show_desc:
        st.subheader("Data Description")
        st.write(sales_df.describe())

# ---- Time Features ----
sales_df['Month'] = sales_df['Date'].dt.month
sales_df['Year'] = sales_df['Date'].dt.year
sales_df['Week'] = sales_df['Date'].dt.isocalendar().week

# ---- Sales Over Time ----
st.subheader("Weekly Sales Over Time")
import streamlit as st
import plotly.express as px
# Group sales by date
sales_by_date = sales_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
# Create interactive line chart
fig = px.line(
    sales_by_date,
    x='Date',
    y='Weekly_Sales',
    title='Total Weekly Sales',
    labels={'Weekly_Sales': 'Weekly Sales', 'Date': 'Date'}
)
# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)

# ---- Holiday Impact ----
st.subheader("Holiday Impact on Sales")
# Convert Holiday_Flag to readable labels
sales_df['Holiday_Label'] = sales_df['Holiday_Flag'].map({0: 'Non-Holiday', 1: 'Holiday'})
# Create interactive box plot
fig = px.box(
    sales_df,
    x='Holiday_Label',
    y='Weekly_Sales',
    title='Holiday vs Non-Holiday Sales',
    labels={'Holiday_Label': 'Holiday', 'Weekly_Sales': 'Weekly Sales'}
)
# Display the interactive chart
st.plotly_chart(fig, use_container_width=True)

# ---- Prophet Forecast ----
st.subheader("Prophet Forecast (12 weeks)")
# Prepare data
weekly_sales = sales_df.groupby('Date')['Weekly_Sales'].sum().sort_index()
df_prophet = weekly_sales.reset_index().rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
# Fit model
model_prophet = Prophet()
model_prophet.fit(df_prophet)
# Forecast
future = model_prophet.make_future_dataframe(periods=12, freq='W')
forecast = model_prophet.predict(future)
# Interactive plot with Plotly
fig_prophet = plot_plotly(model_prophet, forecast)
st.plotly_chart(fig_prophet)

# ----  Sales Forecast Confidence Intervals ----
st.subheader(" Prophet Forecast with Confidence Intervals")

# --- Select confidence interval ---
interval_width = st.selectbox("Select confidence interval width:", [0.80, 0.90, 0.95], index=1)
# Prepare data
df=sales_df
weekly_sales = df.groupby('Date')['Weekly_Sales'].sum().sort_index()
weekly_sales = weekly_sales.reset_index().rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
# Fit Prophet with selected interval
model = Prophet(interval_width=interval_width)
model.fit(weekly_sales)
# Forecast
future = model.make_future_dataframe(periods=12, freq='W-FRI')
forecast = model.predict(future)
# Merge for display
merged = pd.merge(weekly_sales, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='outer')
# Plot
fig = go.Figure()
# Actual
fig.add_trace(go.Scatter(
    x=merged['ds'], y=merged['y'], mode='lines+markers',
    name='Actual Sales', line=dict(color='black')
))
# Forecast
fig.add_trace(go.Scatter(
    x=merged['ds'], y=merged['yhat'], mode='lines',
    name='Forecast', line=dict(color='blue')
))
# Confidence band
fig.add_trace(go.Scatter(
    x=pd.concat([merged['ds'], merged['ds'][::-1]]),
    y=pd.concat([merged['yhat_upper'], merged['yhat_lower'][::-1]]),
    fill='toself',
    fillcolor='rgba(135, 206, 250, 0.3)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    name=f'{int(interval_width*100)}% Confidence Interval'
))
fig.update_layout(
    title=f'Forecast with {int(interval_width*100)}% Confidence Interval',
    xaxis_title='Date',
    yaxis_title='Weekly Sales',
    template='plotly_white'
)
st.plotly_chart(fig)

# --- Download CSV ---
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
st.download_button(
    label=" Download Forecast with Confidence Intervals",
    data=csv,
    file_name=f"forecast_confidence_{int(interval_width*100)}.csv",
    mime="text/csv"
)

# ---- Anomaly Detection ----
st.subheader("Sales Anomaly Detection")
sales_df = sales_df.sort_values('Date')
sales_df['Rolling_Mean'] = sales_df['Weekly_Sales'].rolling(4).mean()
sales_std = sales_df['Weekly_Sales'].std()
sales_df['Anomaly'] = sales_df['Weekly_Sales'] > (sales_df['Rolling_Mean'] + 2 * sales_std)
anomalies = sales_df[sales_df['Anomaly']]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=sales_df['Date'], y=sales_df['Weekly_Sales'], name='Sales', mode='lines'))
fig3.add_trace(go.Scatter(x=sales_df['Date'], y=sales_df['Rolling_Mean'], name='4-Week Avg'))
fig3.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Weekly_Sales'],
                            mode='markers', name='Anomalies', marker=dict(color='red', size=10)))
fig3.update_layout(title='Anomaly Detection', template='plotly_white')
st.plotly_chart(fig3, use_container_width=True)

# ---- Holiday Impact Section ----
st.subheader(" Impact of Holidays on Weekly Sales")
df = sales_df.copy()
df['Holiday_Flag'] = df['Holiday_Flag'].map({0: 'Non-Holiday', 1: 'Holiday'})
# Interactive Plotly box plot
fig = px.box(
    df,
    x='Holiday_Flag',
    y='Weekly_Sales',
    title='Impact of Holidays on Weekly Sales',
    labels={'Holiday_Flag': '', 'Weekly_Sales': 'Weekly Sales'},
    points='all'  # Optional: shows all individual points
)
st.plotly_chart(fig)
# Store Heatmap Section
import plotly.graph_objects as go

# ---- Store Heatmap Section ----
st.subheader("Monthly Sales Heatmap per Store")
df = sales_df.copy()
df['Month'] = df['Date'].dt.to_period('M').astype(str)
store_monthly = df.groupby(['Store', 'Month'])['Weekly_Sales'].sum().reset_index()
# Create pivot table
pivot = store_monthly.pivot(index='Store', columns='Month', values='Weekly_Sales').fillna(0)
# Create interactive heatmap
fig = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=pivot.columns,
    y=pivot.index,
    colorscale='YlGnBu',
    colorbar=dict(title='Sales')
))
fig.update_layout(
    title='Heatmap of Monthly Sales per Store',
    xaxis_title='Month',
    yaxis_title='Store',
    height=600
)
st.plotly_chart(fig)
# Prediction for each store
st.subheader("Select a store for prediction")
store_ids = sorted(df['Store'].unique())
store_id = st.selectbox("Choose a store to highlight (optional):", list(store_ids))
if store_id != "All Stores":
    store_data = df[df['Store'] == store_id]
    weekly_sales = store_data.groupby('Date')['Weekly_Sales'].sum().reset_index()

    if weekly_sales.dropna().shape[0] < 2:
        st.warning(f"Not enough data for Store {store_id} to forecast.")
    else:
        df_prophet = weekly_sales.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=12, freq='W-FRI')
        forecast = model.predict(future)

        merged = pd.merge(df_prophet, forecast[['ds', 'yhat']], on='ds', how='outer')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged['ds'], y=merged['y'], name='Actual Sales'))
        fig.add_trace(go.Scatter(x=merged['ds'], y=merged['yhat'], name='Forecasted Sales (Prophet)'))

        fig.update_layout(
            title=f"Store {store_id} - Weekly Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Weekly Sales",
            template="plotly_white"
        )

        st.plotly_chart(fig)

        csv = merged.to_csv(index=False)
        st.download_button(
            "Download Forecast Data as CSV",
            csv,
            f"forecast_store_{store_id}.csv",
            "text/csv"
        )
# Store Clustering with KMeans
st.subheader("Store Clustering (Unsupervised Learning)")
# Feature Engineering
cluster_df = df.copy()
cluster_df['Holiday_Flag'] = cluster_df['Holiday_Flag'].astype(bool)
# Total and average sales per store
agg = cluster_df.groupby('Store').agg(
    Total_Sales=('Weekly_Sales', 'sum'),
    Avg_Weekly_Sales=('Weekly_Sales', 'mean'),
)
# Holiday sales
holiday_sales = cluster_df[cluster_df['Holiday_Flag']].groupby('Store')['Weekly_Sales'].mean()
non_holiday_sales = cluster_df[~cluster_df['Holiday_Flag']].groupby('Store')['Weekly_Sales'].mean()
# Add to DataFrame
agg['Avg_Holiday_Sales'] = holiday_sales
agg['Avg_NonHoliday_Sales'] = non_holiday_sales
agg = agg.fillna(0)  # in case some stores lack holiday data
# Holiday impact as uplift
agg['Holiday_Impact'] = agg['Avg_Holiday_Sales'] - agg['Avg_NonHoliday_Sales']
# Normalize features
features = ['Total_Sales', 'Avg_Weekly_Sales', 'Holiday_Impact']
X = agg[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# KMeans clustering
k = st.slider("Select number of clusters (K):", min_value=2, max_value=6, value=3)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
agg['Cluster'] = kmeans.fit_predict(X_scaled)
# Interactive Scatter Plot
fig = px.scatter(
    agg.reset_index(),
    x='Total_Sales',
    y='Holiday_Impact',
    color='Cluster',
    hover_data=['Store', 'Avg_Weekly_Sales'],
    title="KMeans Clustering of Stores",
    labels={'Total_Sales': 'Total Sales', 'Holiday_Impact': 'Holiday Impact'}
)
st.plotly_chart(fig, use_container_width=True)
# Show cluster table
if st.checkbox("Show cluster summary table"):
    st.dataframe(agg.reset_index()) 
# Feature Correlation Heatmap
    import plotly.express as px

# ---- Feature Correlation Heatmap ----
st.subheader("Feature Correlation Heatmap")
# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])
# Drop constant columns
numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]
# Compute correlation matrix
corr = numeric_df.corr()
# Create interactive heatmap
fig = px.imshow(
    corr,
    text_auto='.2f',
    color_continuous_scale='RdBu',
    title='Correlation Between Numeric Features',
    aspect='auto'
)
st.plotly_chart(fig)

# ---- Price Sensitivity Analysis ----
st.subheader(" Price Sensitivity Analysis (Interactive with Regression)")
def price_sidebar_filters(df):
    st.sidebar.subheader(" Filter Ranges")
    cpi_min, cpi_max = float(df['CPI'].min()), float(df['CPI'].max())
    fuel_min, fuel_max = float(df['Fuel_Price'].min()), float(df['Fuel_Price'].max())

    cpi_range = st.sidebar.slider(
        "CPI Range", min_value=cpi_min, max_value=cpi_max,
        value=(cpi_min, cpi_max), step=0.5
    )
    fuel_range = st.sidebar.slider(
        "Fuel Price Range", min_value=fuel_min, max_value=fuel_max,
        value=(fuel_min, fuel_max), step=0.1
    )
    holiday_choice = st.sidebar.radio(
        "Filter by Holiday Weeks?", ["All", "Holiday Only", "Non-Holiday"]
    )
    return cpi_range, fuel_range, holiday_choice
# Apply filters
cpi_range, fuel_range, holiday_choice = price_sidebar_filters(df)
filtered_df = df[
    (df['CPI'] >= cpi_range[0]) & (df['CPI'] <= cpi_range[1]) &
    (df['Fuel_Price'] >= fuel_range[0]) & (df['Fuel_Price'] <= fuel_range[1])
].dropna(subset=['Weekly_Sales', 'CPI', 'Fuel_Price'])
if holiday_choice == "Holiday Only":
    filtered_df = filtered_df[filtered_df['Holiday_Flag'] == 1]
elif holiday_choice == "Non-Holiday":
    filtered_df = filtered_df[filtered_df['Holiday_Flag'] == 0]
# Linear Regression
if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your criteria.")
else:
    X = filtered_df[['CPI', 'Fuel_Price']]
    y = filtered_df['Weekly_Sales']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    coef_cpi = model.coef_[0]
    coef_fuel = model.coef_[1]
    intercept = model.intercept_

    st.markdown(f"""
    **Regression Equation:**  
    > Weekly Sales = {intercept:.2f} + {coef_cpi:.2f} × CPI + {coef_fuel:.2f} × Fuel Price

    **R² Score:** {r2:.3f}
    """)


# --- Interactive Scatter Plots with Trendlines ---
fig1 = px.scatter(
    filtered_df, x='CPI', y='Weekly_Sales', trendline='ols',
    opacity=0.6, title='CPI vs Weekly Sales'
)
st.plotly_chart(fig1)
fig2 = px.scatter(
    filtered_df, x='Fuel_Price', y='Weekly_Sales', trendline='ols',
    opacity=0.6, title='Fuel Price vs Weekly Sales'
)
st.plotly_chart(fig2)


# --- Interactive 3D Regression Surface ---
cpi_vals = np.linspace(cpi_range[0], cpi_range[1], 20)
fuel_vals = np.linspace(fuel_range[0], fuel_range[1], 20)
cpi_grid, fuel_grid = np.meshgrid(cpi_vals, fuel_vals)
sales_pred_grid = model.predict(np.column_stack([cpi_grid.ravel(), fuel_grid.ravel()]))
sales_pred_grid = sales_pred_grid.reshape(cpi_grid.shape)

fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(
    x=filtered_df['CPI'],
    y=filtered_df['Fuel_Price'],
    z=filtered_df['Weekly_Sales'],
    mode='markers',
    marker=dict(size=4, color='blue'),
    name='Actual Sales'
))
fig3d.add_trace(go.Surface(
    x=cpi_grid,
    y=fuel_grid,
    z=sales_pred_grid,
    colorscale='Viridis',
    opacity=0.7,
    name='Regression Surface',
    showscale=False
))
fig3d.update_layout(
    title=' Weekly Sales vs CPI and Fuel Price (Interactive)',
    scene=dict(
        xaxis_title='CPI',
        yaxis_title='Fuel Price',
        zaxis_title='Weekly Sales'
    ),
    height=600
)
st.plotly_chart(fig3d)

# --- Prediction Tool ---
st.markdown("### Predict Sales Based on CPI & Fuel Price")
input_cpi = st.number_input("Enter CPI:", value=round(cpi_range[1], 2))
input_fuel = st.number_input("Enter Fuel Price:", value=round(fuel_range[1], 2))

if st.button("Predict Weekly Sales"):
    predicted_sales = model.predict([[input_cpi, input_fuel]])[0]
    st.success(f" Predicted Weekly Sales: ${predicted_sales:,.2f}")





