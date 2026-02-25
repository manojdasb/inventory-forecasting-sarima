import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("📦 Inventory Forecasting using SARIMA")

st.write("Upload a CSV file with Date and Inventory columns.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Convert Date column
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    data = data.asfreq("ME")

    st.subheader("📊 Historical Data")
    st.line_chart(data["Inventory"])

    forecast_period = st.slider("Select Forecast Months", 1, 24, 12)

    if st.button("Run Forecast"):

        model = SARIMAX(
            data["Inventory"],
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        model_fit = model.fit()

        forecast = model_fit.forecast(steps=forecast_period)

        forecast_index = pd.date_range(
            start=data.index[-1],
            periods=forecast_period + 1,
            freq="ME"
        )[1:]

        forecast_df = pd.DataFrame({
            "Date": forecast_index,
            "Forecasted Inventory": forecast.values
        })

        forecast_df.set_index("Date", inplace=True)

        st.subheader("📈 Forecast Results")
        st.write(forecast_df)

        fig, ax = plt.subplots()
        ax.plot(data.index, data["Inventory"], label="Actual")
        ax.plot(forecast_df.index, forecast_df["Forecasted Inventory"], label="Forecast")
        ax.legend()
        ax.set_title("Inventory Forecast")
        st.pyplot(fig)

        csv = forecast_df.to_csv().encode("utf-8")
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name="inventory_forecast.csv",
            mime="text/csv",
        )