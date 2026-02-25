📦 Inventory Demand Forecasting using SARIMA

📌 Project Overview
Accurate inventory forecasting is critical for supply chain efficiency and cost optimization.
This project builds a Seasonal ARIMA (SARIMA) time series forecasting model to predict future inventory demand using historical sales data.

The solution helps businesses:
📉 Reduce stockouts
📦 Minimize overstocking
📊 Improve demand planning
💰 Optimize operational costs
🎯 Business Objective

Many businesses struggle with fluctuating demand patterns due to:
Seasonality
Trend variations
Random demand spikes
Market uncertainty

This project applies statistical time series modeling to generate reliable demand forecasts and support data-driven inventory decisions.
🧠 Model Used
🔹 SARIMA (Seasonal AutoRegressive Integrated Moving Average)

SARIMA extends ARIMA by incorporating seasonality components:
AR (AutoRegression)
I (Integration / Differencing)
MA (Moving Average)
Seasonal parameters (P, D, Q, s)
Model tuning was performed using:
ADF Test (Stationarity check)
ACF & PACF plots

Parameter selection based on statistical evaluation

🛠️ Tech Stack
Python
Pandas
NumPy
Matplotlib
Statsmodels
Scikit-learn

📂 Project Structure
inventory-forecasting-sarima/
│
├── arima_inventory_simple.py
├── simple_requirements.txt
├── README.md
└── data/
