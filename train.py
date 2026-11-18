# train.py - Model training, evaluation, comparison, and saving
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_save_model(X_train, y_train, X_val, y_val, model_path="best_solar_model.pkl"):
    # 1. Baseline Model: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict on validation set and clip negatives to 0
    lr_pred = np.clip(lr.predict(X_val), 0, None)
    lr_mae = mean_absolute_error(y_val, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))
    lr_r2 = r2_score(y_val, lr_pred)
    
    results = []
    results.append({
        "Model": "LinearRegression (Baseline)",
        "n_estimators": "-",
        "max_depth": "-",
        "MAE": round(lr_mae, 2),
        "RMSE": round(lr_rmse, 2),
        "R2": round(lr_r2, 4)
    })
    
    # 2. Main Model: Random Forest Regressor tuning loop
    best_model = None
    best_rmse = float("inf")
    best_setting = ""
    
    for n_est in [50, 100, 150]:
        for depth in [5, 10, None]:
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42)
            rf.fit(X_train, y_train)
            
            rf_pred = np.clip(rf.predict(X_val), 0, None)
            mae = mean_absolute_error(y_val, rf_pred)
            rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
            r2 = r2_score(y_val, rf_pred)
            
            results.append({
                "Model": "RandomForestRegressor",
                "n_estimators": n_est,
                "max_depth": str(depth),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2": round(r2, 4)
            })
            
            # Select model with the lowest validation RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = rf
                best_setting = f"n_estimators={n_est}, max_depth={depth}"
                
    # Display comparison table
    metrics_df = pd.DataFrame(results)
    print("=== Model Performance on Validation Set ===")
    print(metrics_df.to_string(index=False))
    
    # Save best model artifact to disk
    joblib.dump(best_model, model_path)
    print(f"\nBest Model ({best_setting}) saved to '{model_path}'")
    
    return best_model, metrics_df
