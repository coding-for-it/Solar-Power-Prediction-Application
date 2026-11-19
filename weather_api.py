# weather_api.py - Fetch 48-hour weather forecast for Pune from Open-Meteo
import requests
import time
import pandas as pd
from datetime import datetime, timedelta

def get_pune_weather_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 18.52,
        "longitude": 73.86,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,precipitation",
        "wind_speed_unit": "ms",
        "timezone": "Asia/Kolkata",
        "forecast_days": 3
    }
    
    forecast_df = None
    is_fallback = False
    
    # Retry the request up to 3 times
    for attempt in range(1, 4):
        try:
            response = requests.get(url, params=params, timeout=10)
            # Treat 429 rate limit or any non-200 status as a failure to retry
            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})
                
                times = pd.to_datetime(hourly.get("time", []))
                n_hours = len(times)
                
                # Build DataFrame
                forecast_df = pd.DataFrame({
                    "timestamp": times,
                    "temperature": hourly.get("temperature_2m", [25.0] * n_hours),
                    "humidity": hourly.get("relative_humidity_2m", [50.0] * n_hours),
                    "cloud_cover": hourly.get("cloud_cover", [20.0] * n_hours),
                    "wind_speed": hourly.get("wind_speed_10m", [5.0] * n_hours),
                    "precipitation": hourly.get("precipitation", [0.0] * n_hours)
                })
                
                # Fill any None values inside the API lists
                forecast_df = forecast_df.interpolate().fillna(0)
                
                # Keep only rows from the current hour onward and take the first 48
                now = pd.Timestamp.now().floor("h")
                forecast_df = forecast_df[forecast_df["timestamp"] >= now].head(48).reset_index(drop=True)
                
                if len(forecast_df) == 48:
                    print("Successfully fetched live 48-hour forecast from Open-Meteo API.")
                    return forecast_df, False
                else:
                    print(f"Attempt {attempt}: Received fewer than 48 future hours.")
            else:
                print(f"Attempt {attempt}: API returned status code {response.status_code} (retrying...)")
        except Exception as err:
            print(f"Attempt {attempt} failed: {err}")
            
        if attempt < 3:
            time.sleep(2)
            
    # If all 3 attempts failed, use fallback data
    print("WARNING: Open-Meteo API unreachable after 3 attempts. Using fallback forecast data.")
    is_fallback = True
    now = pd.Timestamp.now().floor("h")
    fallback_times = [now + timedelta(hours=i) for i in range(48)]
    forecast_df = pd.DataFrame({
        "timestamp": fallback_times,
        "temperature": [25.0] * 48,
        "humidity": [50.0] * 48,
        "cloud_cover": [20.0] * 48,
        "wind_speed": [5.0] * 48,
        "precipitation": [0.0] * 48
    })
    return forecast_df, is_fallback
