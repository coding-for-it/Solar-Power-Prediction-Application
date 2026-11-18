# preprocess.py - Data cleaning and feature engineering functions
import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    # 1. Read CSV file
    df = pd.read_csv(file_path)
    
    # 2. Parse timestamp column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 3. Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 4. Interpolate missing values and fill boundary gaps
    num_cols = ["temperature", "humidity", "cloud_cover", "wind_speed", "precipitation", "solar_output"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear")
            df[col] = df[col].bfill().ffill()
            
    # 5. Clip values within physical boundaries
    df["humidity"] = df["humidity"].clip(0, 100)
    df["cloud_cover"] = df["cloud_cover"].clip(0, 100)
    df["wind_speed"] = df["wind_speed"].clip(lower=0)
    df["precipitation"] = df["precipitation"].clip(lower=0)
    if "solar_output" in df.columns:
        df["solar_output"] = df["solar_output"].clip(lower=0)
        
    return df

def add_features(df):
    data = df.copy()
    
    # Ensure timestamp is datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    
    # Hour of day (0-23)
    data["hour"] = data["timestamp"].dt.hour
    
    # Cyclical hour encoding (sin and cos)
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24)
    
    # Day of week (0=Monday, 6=Sunday)
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    
    # Weekend indicator (1 if Saturday or Sunday, else 0)
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    
    return data
