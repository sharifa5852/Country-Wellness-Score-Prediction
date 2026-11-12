# Phase 3 experiment completion report

- Checks passed: 26/26
- Random split: 10 seeds x 4 models; mean test RMSE Persistence 0.1905, Linear Regression 0.1832, Random Forest 0.2006, XGBoost 0.1972
- Rolling origin: evaluated years [2019, 2020, 2021, 2022, 2023, 2024]; skipped [2018]
- Rolling mean RMSE: Persistence 0.1850, Linear Regression 0.1820, Random Forest 0.2094, XGBoost 0.2021
- SHAP model: Random Forest (chosen on chronological validation); top feature HappinessScore (97.3% of total mean |SHAP|)
- Not repeated in Experiments A/B: LSTM and stacking (time budget).
- All explanations describe model behaviour, not causes.
