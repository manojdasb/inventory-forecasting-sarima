import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# 1. Generate Sample Inventory Data
# -----------------------------
def generate_sample_inventory_data():
    np.random.seed(42)

    dates = pd.date_range('2021-01-01', periods=36, freq='ME')

    trend = np.linspace(200, 400, 36)
    seasonality = 50 * np.sin(np.linspace(0, 6 * np.pi, 36))
    noise = np.random.normal(0, 20, 36)

    inventory = trend + seasonality + noise

    data = pd.DataFrame({
        "Date": dates,
        "Inventory": inventory
    })

    data.set_index("Date", inplace=True)
    return data


# -----------------------------
# 2. Train SARIMA Model
# -----------------------------
def train_sarima_model(train_data):
    model = SARIMAX(
        train_data,
        order=(1, 1, 1),
        seasonal_order=(1,0,1,12),  # 12 = monthly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit()
    return model_fit


# -----------------------------
# 3. Evaluate Model
# -----------------------------
def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print("\n📊 Model Performance:")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))


# -----------------------------
# 4. Plot Forecast
# -----------------------------
def plot_forecast(data, forecast):

    forecast_index = pd.date_range(
        start=data.index[-1],
        periods=len(forecast) + 1,
        freq='ME'
    )[1:]

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Inventory"], label="Actual Data")
    plt.plot(forecast_index, forecast, label="Forecast")
    plt.legend()
    plt.title("Inventory Forecast (SARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Inventory Level")
    plt.tight_layout()
    plt.savefig("sarima_forecast_plot.png")
    plt.show()


# -----------------------------
# 5. Main Execution
# -----------------------------
def main():

    data = generate_sample_inventory_data()

    train_size = int(len(data) * 0.8)
    train = data[:train_size]
    test = data[train_size:]

    model_fit = train_sarima_model(train["Inventory"])

    predictions = model_fit.forecast(steps=len(test))

    evaluate_model(test["Inventory"], predictions)

    # 12 Month Future Forecast
    future_forecast = model_fit.forecast(steps=12)

    print("\n📈 12-Month Future Forecast:")
    print(future_forecast)

    future_forecast.to_csv("sarima_forecast_results.csv")

    plot_forecast(data, future_forecast)


if __name__ == "__main__":
    main()