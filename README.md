# Solar Power Prediction Application

## Project overview
This project predicts solar power output using weather information for Pune, Maharashtra.
It follows the Embed Square Solutions assignment workflow:
- Generate and load hourly solar-weather data
- Clean and explore the data
- Create time-based and cyclical features
- Train and compare Random Forest, XGBoost and LightGBM regression models
- Select the model with the lowest RMSE
- Try a Voting Regressor only when the best individual model R2 is below 0.85
- Fetch new hourly weather forecasts from Open-Meteo
- Predict future solar output and visualize the results

## Files
- `Solar_Power_Prediction.ipynb` - complete Jupyter Notebook.
- `requirements.txt` - Python packages required by the notebook.
- `solar_weather_dataset.csv` - generated 60-day hourly dataset from the assignment.

The notebook also saves the selected model as `solar_power_model.pkl`, the feature list as `solar_power_features.pkl`, and predictions as `solar_power_predictions.csv` after a successful API call.

## Setup
1. Install Python 3.10 or later.
2. Install Jupyter Notebook or JupyterLab.
3. Run:

```bash
pip install -r requirements.txt
jupyter notebook
```

4. Open `Solar_Power_Prediction.ipynb` and run cells from top to bottom.

## Weather API
The notebook uses the Open-Meteo forecast API: `https://api.open-meteo.com/v1/forecast`.

Location: Pune, Maharashtra, India (Latitude 18.52, Longitude 73.86).
The notebook requests hourly temperature, relative humidity, cloud cover, wind speed and precipitation for the next 48 available forecast hours. No API key is required for normal non-commercial use. Internet access is required. If the API is unavailable, the notebook displays a message and skips future predictions rather than stopping the entire notebook.

## Data processing and features
The assignment's provided data-generation formula creates 1,440 hourly rows from 1 January 2024 through 29 February 2024. This dataset is synthetic.

The notebook parses timestamps, checks missing values and duplicates, fills missing numeric values with the median, and inspects numerical distributions with boxplots. Outliers are not automatically deleted because extreme values can be valid observations.

Features include temperature, humidity, cloud cover, wind speed, precipitation, hour, day, month, day of week, weekend flag, and sine/cosine hour features.

## Model evaluation
The notebook uses an 80:20 shuffled train-test split with `random_state=42`. It compares models using MAE, RMSE and R2. Lower MAE/RMSE and higher R2 are preferred. The model with the lowest RMSE is selected.

### Reference run (environment-dependent)
A validation run produced these reference results:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| XGBoost | 14.46 | 18.75 | 0.9864 |
| LightGBM | 15.51 | 20.54 | 0.9837 |
| Random Forest | 16.43 | 22.69 | 0.9802 |

Run the notebook on your computer and use the values printed by your run as the final reported metrics. Small differences may occur across library versions.

## Important assumptions and limitations
- The requested shuffled split is used. Because this is time-related data, a shuffled split can place nearby hours in both training and testing sets and may give an optimistic score. A chronological holdout is more representative of future forecasting and is recommended for production evaluation.
- The historical dataset is synthetic and covers January-February 2024. The future forecast is fetched for the current date. They are not a continuous real-world plant history; the combined chart demonstrates the workflow only.
- The generated target is based on the assignment's synthetic formula. Strong scores on this dataset do not prove accuracy on real solar-plant measurements.
- Predictions are clipped at zero because solar output cannot be negative.
- Weather API access requires an internet connection.

## Final run checklist
- Run all notebook cells from top to bottom.
- Confirm that all three models train and the comparison table appears.
- Confirm that Open-Meteo returns future hourly data (requires internet).
- Confirm that the future prediction table and final chart appear.
- Save the notebook with its outputs before submission.
- Capture a screenshot of the final historical + forecast chart for submission.
