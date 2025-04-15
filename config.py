DATA_DIR = "data/Data/Data.csv" #replace with path to data
RESULTS_DIR = "results/result.csv"
CLIMATE_DATA_PATH = "data/Climate Data/Climate Data/Kalam Climate Data.xlsx" #replace with path to climate data
SUBMISSION_FILE_PATH = "data/SampleSubmission.csv" #replace with submission file path
RANDOM_STATE = 42
N_FOLDS = 5
TARGET = "kwh"
PARAMS = {
    "boosting_type": 'gbdt',"num_leaves": 77, "max_depth": 7, "colsample_bytree": 0.45, "learning_rate": 0.45,
    'min_child_samples': 20,'min_split_gain': 0.45, "n_estimators": 200,"verbose": -1, "metric": "rmse",
    "force_col_wise": True
    }
FEATURES = {
    'f1': 'temp_feature_range/dew_point_depression/snowfall_sum/is_snowy/v_blue/kwh'.split('/'),
    'f2': 'dew_mean/temp_mean/month/kwh'.split('/'), 'f3': 'month/days/kwh'.split('/')
    }
WINDOWS = 4 #[4, 6, 15, 17]