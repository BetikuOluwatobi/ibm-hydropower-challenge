---

# Micro-Hydro Power Load Forecasting Solution

## Competition Overview
This project was developed for the IBM SkillsBuild/Zindi Africa "Forecasting Climate and Operational Effects on Load Generation for Micro-Hydropower Plants" competition. The challenge focused on predicting energy load generation (in kWh) for off-grid micro-hydropower plants in Pakistan, using operational data (voltage, current, power factors) combined with climate indicators (temperature, dew point, wind speed, precipitation).

## Solution Approach
### Key Features
- **Hybrid Feature Engineering**: Combined time-series patterns with climate indicators
- **Advanced Windowed Forecasting**: Implemented rolling window feature generation
- **Ensemble Modeling**: Used LightGBM with VotingRegressor for robust predictions
- **Climate Feature Synthesis**: Created derived weather metrics (apparent temp, wind direction, etc.)

### Technical Implementation
1. **Data Processing Pipeline**:
   - Time-based aggregation of power metrics
   - Climate data feature engineering
   - Anomalous user filtering
   - Phase detection (1-phase vs 3-phase systems)

2. **Feature Engineering**:
   - Rolling window statistics (4, 6, 15, 17-day windows)
   - Climate-derived features:
     - Dew point depression
     - Apparent temperature
     - Wind speed/direction
     - Snowfall indicators

3. **Model Architecture**:
   - LightGBM ensemble with multiple random seeds
   - MinMax scaling pipeline
   - K-Fold cross-validation (n_splits=5)
   - Parallelized training implementation

4. **Forecasting Method**:
   - Recursive prediction with climate feature integration
   - User-specific phase consideration
   - Multi-window ensemble averaging

## Performance
- **Public Leaderboard**: 6.808 RMSE (Rank 100/444)
- **Private Leaderboard**: 5.618 RMSE

  
![BetikuOluwatobi-IBM SkillsBuild Hydropower Climate Optimisation Challenge ](https://github.com/user-attachments/assets/9397703c-a1d2-4898-8335-12d160edf7bf)


