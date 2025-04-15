# SUPPRESS ALL WARNINGS FIRST
import warnings, logging, os
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('sklearnex').setLevel(logging.WARNING)

# import libraries
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
from src.preprocessing import preprocess_data, generate_climate
from config import DATA_DIR, CLIMATE_DATA_PATH, PARAMS, RESULTS_DIR, RANDOM_STATE, SUBMISSION_FILE_PATH, N_FOLDS, TARGET, FEATURES, WINDOWS 
from src.forecast import generate_predictions

# PATCH SCIKIT-LEARN (Silently)
patch_sklearn(verbose=False)
if __name__ == "__main__":
    os.environ["LIGHTGBM_VERBOSE"] = "0"
    # Generate forecast
    data = pd.read_csv(DATA_DIR)
    climate = pd.read_excel(CLIMATE_DATA_PATH)
    submission = pd.read_csv(SUBMISSION_FILE_PATH)
    processed_data = preprocess_data(data, target=TARGET, agg='sum', cutoff_date='2023-07-01')
    climate_data = generate_climate(climate)
    result = generate_predictions(
        data=processed_data, windows=WINDOWS, ss=submission, params=PARAMS, climate_data=climate_data, 
        cols=FEATURES, n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    result.to_csv(RESULTS_DIR, index=False)