import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from src.preprocessing import RollingHistory
from modelling.model import make_lgb, generate_forecast


def predict_columns(data, grouped, idx, windows, cols, unique_ids, params, climate_data,
                    n_splits, shuffle, random_state):
    user_data = grouped.get_group(idx)
    start_date = user_data['date'].max() + pd.Timedelta(days=1)
    phase = user_data['v_blue'].iloc[0]
    forecast_indices = pd.date_range(start=start_date, periods=31, freq='D')

    if type(windows) == list:
        predictions = predict_windows(data, user_data, windows, cols, unique_ids, forecast_indices, params, climate_data, 
                                      phase, n_splits, shuffle=shuffle, random_state=random_state)
    else:
        history = user_data[cols].values.flatten()
        rolling_tf = RollingHistory(window=windows, unique_ids=unique_ids, cols=cols, data=user_data)
        windowed_data, targets = rolling_tf.fit_transform(data) # windowed_data, targets = rolling_window(user_data, window=windows, cols=cols)
        predictions = predict_kfold(windowed_data, targets, history, forecast_indices, params, cols, climate_data, 
                                    phase, n_splits, shuffle=shuffle, random_state=random_state)

    return forecast_indices, predictions

def predict_features(data, grouped, idx, windows, columns, unique_ids, params, climate_data,
                     n_splits, shuffle, random_state):
    user_data = grouped.get_group(idx)
    phase = user_data['v_blue'].iloc[0]
    start_date = user_data['date'].max() + pd.Timedelta(days=1)
    forecast_indices = pd.date_range(start=start_date, periods=31, freq='D')
    predictions = np.zeros(len(forecast_indices))
    num_cols = len(columns.items())
    
    for _, cols in columns.items():
        if type(windows) == list:
            preds = predict_windows(data, user_data, windows, cols, unique_ids, forecast_indices, params, climate_data, 
                                    phase, n_splits, shuffle=shuffle, random_state=random_state)
        else:
            history = user_data[cols].values.flatten()
            rolling_tf = RollingHistory(window=windows, unique_ids=unique_ids, cols=cols, data=user_data)
            windowed_data, targets = rolling_tf.fit_transform(data) # windowed_data, targets = rolling_window(user_data, window=windows, cols=cols)
            preds = predict_kfold(windowed_data, targets, history, forecast_indices, params, cols, climate_data, 
                                  phase, n_splits, shuffle=shuffle, random_state=random_state)
        predictions += preds
        
    return forecast_indices, predictions/num_cols

def predict_windows(data, user_data, windows, cols, unique_ids, forecast_indices, params, climate_data, 
                    phase, n_splits=5, shuffle=True, random_state=123):
    all_predictions = []
    history = user_data[cols].values.flatten()
    
    for window in windows:
        rolling_tf = RollingHistory(window=window, unique_ids=unique_ids, cols=cols, data=user_data) # windowed_data, targets = rolling_window(user_data, window=window, cols=cols)
        windowed_data, targets = rolling_tf.fit_transform(data)
        preds = predict_kfold(windowed_data, targets, history, forecast_indices, params, cols, climate_data, phase, 
                              n_splits, shuffle=shuffle, random_state=random_state)
        all_predictions.append(preds)
    all_predictions = np.array(all_predictions)
    return np.mean(all_predictions, axis=0)

def predict_kfold(windowed_data, targets, history, forecast_indices, params, cols, climate_data, 
                  phase, n_splits=5, shuffle=True, random_state=123):
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    model = make_lgb(params=params, seeds=[234, 262, 342, 408])

    def train_fold(train_idx, test_idx):
        """Train and evaluate a single fold in parallel."""
        X_train, X_test = windowed_data[train_idx], windowed_data[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        
        model.fit(X_train, y_train)

        # Generate predictions
        y_preds = model.predict(X_test)
        preds = generate_forecast(model=model, indices=forecast_indices, window=X_train.shape[-1], 
                                  history=history, climate_data=climate_data, features=cols, phase=phase)

        return np.array(preds)

    # Run K-Fold in parallel
    results = Parallel(n_jobs=-1, backend="loky")(delayed(train_fold)(train_idx, test_idx) for train_idx, test_idx in kfold.split(windowed_data))
    return sum(res for res in results) / n_splits