import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin

def feature_range(group):
    return group.max() - group.min()

def set_phase(row):
    return 3 if row.mean() > 0 else 1

def generate_climate(data):
    data = data.rename(columns={'Temperature (°C)': "temp", 'Dewpoint Temperature (°C)': "dew", 
                                'U Wind Component (m/s)': "u_wind", 'V Wind Component (m/s)': 'v_wind', 
                                'Total Precipitation (mm)': 'total_prep', 'Snowfall (mm)': 'snowfall'})
    data["date"] = data['Date Time'].dt.date
    data = data.groupby('date').agg({'temp': ['mean', feature_range], 'dew': 'mean', 'u_wind': 'mean', 'v_wind': 'mean', 
                                     'total_prep': 'sum','snowfall': 'sum'}).reset_index()
    data.columns = ['_'.join(col) if col[1] else col[0] for col in data.columns]
    data["dew_point_depression"] = data["temp_mean"] - data["dew_mean"]
    data["apparent_temp"] = data["temp_mean"] + 0.33 * data["dew_mean"] - 0.7 * (data["u_wind_mean"]**2 + data["v_wind_mean"]**2)**0.5 - 4.0
    data["wind_speed"] = (data["u_wind_mean"]**2 + data["v_wind_mean"]**2)**0.5
    data["wind_direction"] = np.degrees(np.arctan2(data["v_wind_mean"], data["u_wind_mean"])) % 360
    data["is_snowy"] = (data["snowfall_sum"] > 0).astype(int)
    return data.set_index('date')

def preprocess_data(df, target='kwh', agg='sum', cutoff_date='2024-07-01'):
    filter_user = ['consumer_device_11','consumer_device_14', 'consumer_device_15' 'consumer_device_17', 'consumer_device_24', 'consumer_device_19','consumer_device_20', 
                   'consumer_device_4','consumer_device_6','consumer_device_7','consumer_device_9','consumer_device_33','consumer_device_38', 'consumer_device_39'] 
    
    #'consumer_device_11','consumer_device_14','consumer_device_15',consumer_device_17','consumer_device_24'(optional)
    df = df.fillna({'v_blue': 0})
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['date'] = df['date_time'].dt.date
    data = df.groupby(['Source','date']).agg({'v_blue': set_phase, target: agg, 'consumer_device_x': lambda x: x.mode()}).reset_index().copy()
    data['days'] = pd.to_datetime(data['date']).dt.day
    data['month'] = pd.to_datetime(data['date']).dt.month
    data = data.sort_values(by=['Source','date'], ascending=[True, True])
    data = data[~data['Source'].str.startswith(tuple(filter_user))]
    data = data.loc[pd.to_datetime(data['date']) >= pd.to_datetime(cutoff_date)]
    return data

class RollingHistory(BaseEstimator, TransformerMixin):
    def __init__(self, window, unique_ids, cols, data=None):
        self.unique_ids = set(unique_ids)
        self.cols = cols
        self.window = window
        self.data = data

    def rolling_window_(self, data, window, cols):
        df = data[cols].values 
        size = df.shape[0] - window
        
        if size <= 0:
            return np.empty((0, len(cols) * window + 1))
        
        strides = np.lib.stride_tricks.as_strided(
            df,
            shape=(size, window, len(cols)),
            strides=(df.strides[0], df.strides[0], df.strides[1]),
            writeable=False
        )
        
        data = strides.reshape(size, -1)
        target = df[window:, -1].reshape(-1, 1)
        return np.hstack((data, target))
    
    def fit(self, X, y=None):
        self.history_ids = X.loc[~X['Source'].isin(self.unique_ids), 'Source'].unique()
        return self
        
    def transform(self, X):
        X_sorted = X.sort_values(by=['Source', 'date'])  # Sort once globally
        grouped = X_sorted.groupby('Source')
        histories = [self.rolling_window_(group, self.window, self.cols) for name, group in grouped if name in self.history_ids]

        if self.data is not None:
            histories.append(self.rolling_window_(self.data, self.window, self.cols))

        history = np.concatenate(histories, axis=0)
        return np.array(history[:, :-1]), history[:, -1]

