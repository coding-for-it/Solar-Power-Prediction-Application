import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Project files and model features
# --------------------------------------------------
MODEL_FILE = "solar_power_model.pkl"
FEATURE_FILE = "solar_power_features.pkl"
DATA_FILE = "solar_weather_dataset.csv"

LATITUDE = 18.52
LONGITUDE = 73.86
TIMEZONE = "Asia/Kolkata"

st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="wide"
)


# --------------------------------------------------
# Feature engineering (must match the notebook)
# --------------------------------------------------
def add_features(data):
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    data["hour"] = data["timestamp"].dt.hour
    data["day"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    return data


# --------------------------------------------------
# Load saved model, feature list and historical data
# --------------------------------------------------
@st.cache_resource
def load_project_files():
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURE_FILE)

    history = pd.read_csv(DATA_FILE)
    history["timestamp"] = pd.to_datetime(history["timestamp"])

    numeric_columns = [
        "temperature", "humidity", "cloud_cover",
        "wind_speed", "precipitation", "solar_output"
    ]

    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")
        history[column] = history[column].fillna(history[column].median())

    history = history.drop_duplicates()
    history = history.dropna(subset=["timestamp", "solar_output"])
    history = add_features(history)

    missing_columns = [column for column in features if column not in history.columns]
    if missing_columns:
        raise ValueError(f"Missing model features in dataset: {missing_columns}")

    # Recreate the same shuffled 80:20 split used in the notebook.
    X = history[features]
    y = history["solar_output"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    predictions = np.clip(model.predict(X_test), 0, None)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions)
    }

    return model, features, history, metrics


# --------------------------------------------------
# Get upcoming weather from Open-Meteo
# --------------------------------------------------
@st.cache_data(ttl=600)
def get_weather_forecast():
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "wind_speed_10m",
            "precipitation"
        ],
        "forecast_days": 3,
        "timezone": TIMEZONE
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    weather = response.json()["hourly"]

    forecast = pd.DataFrame({
        "timestamp": pd.to_datetime(weather["time"]),
        "temperature": weather["temperature_2m"],
        "humidity": weather["relative_humidity_2m"],
        "cloud_cover": weather["cloud_cover"],
        "wind_speed": weather["wind_speed_10m"],
        "precipitation": weather["precipitation"]
    })

    forecast = forecast.dropna().reset_index(drop=True)

    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)
    forecast = forecast[forecast["timestamp"] >= now].head(48).copy()

    if forecast.empty:
        raise ValueError("The weather service did not return upcoming hourly data.")

    return forecast


# --------------------------------------------------
# Make predictions for new weather data
# --------------------------------------------------
def make_predictions(model, features, weather_data):
    forecast = add_features(weather_data)
    forecast["predicted_solar_output"] = np.clip(
        model.predict(forecast[features]),
        0,
        None
    )
    forecast["predicted_solar_output"] = forecast["predicted_solar_output"].round(2)
    return forecast


# --------------------------------------------------
# Chart
# --------------------------------------------------
def create_chart(history, forecast, show_history):
    fig = go.Figure()

    if show_history:
        recent_history = history.tail(7 * 24)

        fig.add_trace(go.Scatter(
            x=recent_history["timestamp"],
            y=recent_history["solar_output"],
            mode="lines",
            name="Historical solar output",
            line=dict(width=2)
        ))

    fig.add_trace(go.Scatter(
        x=forecast["timestamp"],
        y=forecast["predicted_solar_output"],
        mode="lines+markers",
        name="Predicted solar output",
        line=dict(width=2, dash="dash"),
        marker=dict(size=5)
    ))

    fig.update_layout(
        title="Solar Power Output",
        xaxis_title="Date and time",
        yaxis_title="Solar output (kW)",
        template="plotly_white",
        hovermode="x unified",
        height=480
    )

    return fig


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Settings")

chart_option = st.sidebar.radio(
    "Chart view",
    ["Past 7 days + next 48 hours", "Next 48 hours only"]
)

table_hours = st.sidebar.selectbox(
    "Hours to display",
    [24, 48]
)

if st.sidebar.button("Refresh weather forecast"):
    get_weather_forecast.clear()


# --------------------------------------------------
# Main application
# --------------------------------------------------
st.title("☀️ Solar Power Prediction Application")
st.write("Hourly solar power predictions for Pune using weather information.")

try:
    model, features, history, metrics = load_project_files()
except FileNotFoundError as error:
    st.error(
        f"Required project file not found: {error.filename}. "
        "Keep app.py, the model, feature file and CSV in the same folder."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the project files: {error}")
    st.stop()

try:
    with st.spinner("Getting the latest weather forecast..."):
        weather_data = get_weather_forecast()
    forecast = make_predictions(model, features, weather_data)
    st.success("Weather forecast loaded successfully from Open-Meteo.")
except Exception as error:
    st.error(f"Could not get the live weather forecast: {error}")
    st.info("Check your internet connection and click 'Refresh weather forecast'.")
    st.stop()


# --------------------------------------------------
# Summary metrics
# --------------------------------------------------
next_24_hours = forecast.head(24)
peak_row = next_24_hours.loc[
    next_24_hours["predicted_solar_output"].idxmax()
]

st.subheader(
    f"Peak predicted output: {peak_row['predicted_solar_output']:.2f} kW"
)
st.caption(f"Expected peak time: {peak_row['timestamp']:%d %b %Y, %H:%M}")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Predicted energy (next 24 hours)",
    f"{next_24_hours['predicted_solar_output'].sum():,.2f} kWh"
)

col2.metric(
    "Productive hours",
    f"{int((next_24_hours['predicted_solar_output'] > 100).sum())} / 24"
)

col3.metric(
    "Average cloud cover",
    f"{next_24_hours['cloud_cover'].mean():.1f}%"
)

col4.metric(
    "Average temperature",
    f"{next_24_hours['temperature'].mean():.1f} °C"
)


# --------------------------------------------------
# Prediction chart
# --------------------------------------------------
st.subheader("Historical and predicted solar output")

show_history = chart_option.startswith("Past")
st.plotly_chart(
    create_chart(history, forecast, show_history),
    use_container_width=True
)


# --------------------------------------------------
# Hourly prediction table
# --------------------------------------------------
st.subheader("Hourly solar power predictions")

table_data = forecast.head(table_hours).copy()

table_data = pd.DataFrame({
    "Date and time": table_data["timestamp"].dt.strftime("%d %b %Y, %H:%M"),
    "Predicted output (kW)": table_data["predicted_solar_output"],
    "Temperature (°C)": table_data["temperature"],
    "Humidity (%)": table_data["humidity"],
    "Cloud cover (%)": table_data["cloud_cover"],
    "Wind speed (m/s)": table_data["wind_speed"],
    "Precipitation (mm)": table_data["precipitation"]
})

st.dataframe(table_data, use_container_width=True, hide_index=True)

download_data = forecast[
    [
        "timestamp",
        "temperature",
        "humidity",
        "cloud_cover",
        "wind_speed",
        "precipitation",
        "predicted_solar_output"
    ]
].copy()

download_data["timestamp"] = download_data["timestamp"].astype(str)

st.download_button(
    label="Download 48-hour predictions as CSV",
    data=download_data.to_csv(index=False),
    file_name="solar_power_predictions.csv",
    mime="text/csv"
)


# --------------------------------------------------
# Model evaluation
# --------------------------------------------------
st.subheader("Model performance")

metric1, metric2, metric3 = st.columns(3)

metric1.metric("MAE", f"{metrics['mae']:.2f}")
metric2.metric("RMSE", f"{metrics['rmse']:.2f}")
metric3.metric("R² score", f"{metrics['r2']:.4f}")

st.caption(
    "These scores are calculated on the same shuffled 20% test split "
    "used in the Jupyter Notebook. Since the project dataset is synthetic, "
    "the results demonstrate the workflow and should not be treated as "
    "real solar-plant performance."
)

st.info(
    "Note: Historical values come from the project's generated dataset. "
    "Upcoming predictions use weather forecast data for Pune. "
    "Actual solar generation can differ from these predictions."
)
