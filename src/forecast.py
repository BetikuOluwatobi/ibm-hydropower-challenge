
import numpy as np
import pandas as pd
import os, tqdm, math, warnings, logging
from src.aggregation import predict_columns, predict_features
from memory_profiler import profile


@profile
def generate_predictions(data, windows, ss, params, climate_data, cols, n_splits=5, shuffle=True, random_state=123):
    # Extract unique IDs more efficiently
    data = data.merge(climate_data, how='inner', on='date')
    unique_ids = set(ss['ID'].str.split('_').str[1:].str.join('_'))
    
    data['date'] = pd.to_datetime(data['date'])  # Ensure date is datetime
    data = data.sort_values(by=['Source', 'date']) 
    grouped = data.groupby('Source')
    
    forecast_results = []
    for idx in tqdm.tqdm(unique_ids, desc="Generating target data...."):
        if isinstance(cols, dict):
            forecast_indices, predictions = predict_features(data=data, grouped=grouped, idx=idx, windows=windows, columns=cols, unique_ids=unique_ids, 
                                                             params=params, climate_data=climate_data, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        else:
            forecast_indices, predictions = predict_columns(data=data, grouped=grouped, idx=idx, windows=windows, cols=cols, unique_ids=unique_ids, 
                                                            params=params, climate_data=climate_data, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        forecast_df = pd.DataFrame({
            'ID': [f"{date.strftime('%Y-%m-%d')}_{idx}" for date in forecast_indices],
            'kwh': predictions
        })
        forecast_results.append(forecast_df)
        print(f"""User: {idx} - Forecast Range: ({max(predictions)} - {min(predictions)})""")
        print()
    
    return pd.concat(forecast_results, ignore_index=True)