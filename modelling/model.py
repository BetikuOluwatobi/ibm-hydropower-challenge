import numpy as np
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline



def make_lgb(params,seeds):
    clfs = []
    for i, seed in enumerate(seeds):
        params["random_state"] = seed
        clfs.append((f"lgb_{i+1}", lgb.LGBMRegressor(**params)))
    
    voting_reg = VotingRegressor(clfs, n_jobs=-1)

    # Wrap in pipeline with MinMaxScaler
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("voting_regressor", voting_reg)
    ])
    
    return pipeline

def generate_forecast(model, indices, window, history, climate_data, features, phase):
    """
        history: Takes train_ds[['month', 'days', 'kwh']].values.flatten() e.g: sample[['days','kwh']].values.flatten()[-10:]
    """
    predictions = np.empty(len(indices), dtype=np.float32)
    for i, dt in enumerate(indices):
        user_climate = climate_data.loc[dt.date()].values
        logit = model.predict(history[-window:].reshape(1, -1)).item()
        predictions[i] = logit  # Store prediction
        feature_values = {
            'temp_mean': user_climate[0], 'temp_feature_range': user_climate[1], 'dew_mean': user_climate[2], 
            'u_wind_mean': user_climate[3], 'v_wind_mean': user_climate[4], 'total_prep_sum': user_climate[5], 
            'snowfall_sum': user_climate[6], 'dew_point_depression': user_climate[7], 'apparent_temp': user_climate[8], 
            'wind_speed': user_climate[9], 'wind_direction': user_climate[10], 'is_snowy': user_climate[11], 
            'v_blue': phase,'month': dt.month, 'days': dt.day, 'kwh': logit
        }
        
        history = np.concatenate((history[-window:], [feature_values[f] for f in features]))
    return predictions