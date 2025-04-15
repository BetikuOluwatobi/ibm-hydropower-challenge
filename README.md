---

# IBM SkillsBuild Micro-Hydro Load Forecasting

Forecasting daily energy consumption (kWh) for off-grid micro-hydropower (MHP) systems using climate and operational data.  
Developed for the [Zindi Africa IBM SkillsBuild Challenge 2025](https://zindi.africa/competitions/ibm-skillsbuild-hydropower-climate-optimisation-challenge), this solution placed **100th out of 444** with a public leaderboard RMSE of **6.8080** and a private leaderboard RMSE of **5.6184**.

## Requirements

- Python 3.11+
- ~8GB RAM recommended (higher if running full forecast mode(Note: Full forecast takes approximately 8 hours, to run lower forecast change windows in config.py to 4))
- Libraries in `requirements.txt`  
- Data files from [Zindi competition page](https://zindi.africa/competitions/ibm-skillsbuild-hydropower-climate-optimisation-challenge/data)

## Repo Structure

```
├── config.py                 # Global configuration paths and constants
├── data/                    # Raw and processed data
│   ├── Climate Data/
│   │   ├── Kalam Climate Data.xlsx
│   └── Data/
│       └── Data.csv
├── modelling/
│   ├── model.py             # Main LightGBM model pipeline and training logic
├── src/
│   ├── preprocessing.py     # Data cleaning and preprocessing steps
│   ├── aggregation.py       # Feature engineering and rolling window aggregations
│   ├── forecast.py          # Prediction generation logic
├── results/                 # Prediction outputs and intermediate files
├── run.py                   # Main pipeline entrypoint (train + predict)
├── requirements.txt
└── README.md
```

## Usage

### 1. Setup Environment
```bash
python -m venv venv
Linux: source venv/bin/activate, Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data
- Place the downloaded `Data.csv`, `SampleSubmission.csv`, and `Kalam Climate Data.xlsx` in their respective subdirectories under `data/`.
- Configure `config.py` if needed to reflect updated paths.

### 3. Run Forecast Pipeline
```bash
python run.py
```

This script:
- Loads and merges climate and MHP operational data.
- Performs preprocessing and rolling window feature engineering.
- Trains LightGBM models using k-fold CV (default: 5 splits).
- Generates predictions per unique user (MHP location) with ensemble smoothing.

## Scripts Overview

| Script        | Purpose                                                  |
|---------------|----------------------------------------------------------|
| `run.py`      | Run full preprocessing, training, and inference pipeline |
| `model.py`    | Build and tune LightGBM forecasting model                |
| `aggregation.py` | Generate rolling window and statistical features     |
| `forecast.py` | Generate user-level time-series forecasts                |

## Acknowledgements

- Data and problem setup by **Zindi Africa** and **IBM SkillsBuild**
- Forecasting models powered by **LightGBM**

## Author
[Betiku Oluwatobi](http://linkedin.com/in/oluwatobi-betiku-oluwatobi/)
