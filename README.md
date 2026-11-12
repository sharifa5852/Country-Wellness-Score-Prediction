# Country Happiness Forecasting

Forecasting each country's **next-year happiness score** (World Happiness Report Cantril ladder, 0-10) from its own history and a few socioeconomic indicators, with a leakage-safe, chronological evaluation. Repository name: `Country-Wellness-Score-Prediction`; "wellness" and "happiness" mean the same quantity here.

**Status:** data pipeline, six models, two evaluation experiments, SHAP analysis and validation checks are complete (Phase 1 audit passed: True; Phase 3 validation: 26/26 checks passed).

## 1. Research question

Can a leakage-safe, chronologically validated, World Bank-enriched forecasting framework (classical ML + LSTM + stacking) predict next-year country happiness better than naive persistence, and how much does random-split evaluation overstate forecasting performance?

## 2. Data

| Item | Source (via Our World in Data) |
|---|---|
| Happiness (Cantril ladder, 0-10) | World Happiness Report |
| GDP per capita (PPP) | World Bank |
| Unemployment rate | World Bank (SL.UEM.TOTL.ZS) |
| Individuals using the internet (%) | World Bank (IT.NET.USER.ZS) |
| Current health expenditure (% of GDP) | WHO Global Health Expenditure Database |

Education spending and the Gini index were evaluated and **excluded** (education spending: 56.1% coverage in the 2023 test year; Gini index: 48.5% overall coverage; see `results/tables/feature_vetting_results.csv` and `results/tables/coverage_by_split.csv`). The raw file covers 159 countries, years 2011-2024; the year 2013 is absent for every country. The data were downloaded once and committed (`data/raw/`); the online sources may be revised later.

## 3. Features and temporal setup

One row = one country at **origin year t**. The target is the happiness score in **t+1**. The 18 model features are five variables (happiness, log GDP, unemployment, internet use, health expenditure) at years t, t-1 and t-2, plus three 0/1 flags marking values that were missing at year t. A row exists only if the real observations for t-2, t-1, t and t+1 all exist (exact calendar-year joins). This replaced an earlier row-position `shift()` approach that misaligned years around the missing 2013 (`results/tables/temporal_alignment_audit_by_year.csv`). Missing covariates are filled only from the same country's earlier real values (at most 2 years back); anything left is imputed with a training-set median. Imputers and scalers are fitted on training rows only. The target is never scaled.

| Split | Origin years | Target years | Rows | Countries |
|---|---|---|---|---|
| train | 2016-2021 | 2017-2022 | 842 | 152 |
| val | 2022-2022 | 2023-2023 | 133 | 133 |
| test | 2023-2023 | 2024-2024 | 133 | 133 |

Total: 1108 samples, 152 countries. Full column descriptions: `data/processed_v2/data_dictionary.md`.

## 4. Models

1. **Persistence**: predicts next year's score = this year's score (the minimum bar).
2. **Linear Regression**: a weighted sum of the 18 scaled features.
3. **Random Forest**: many decision trees averaged (settings chosen on validation: depth 4, min leaf 3).
4. **XGBoost**: trees built one after another to correct earlier errors (depth 2, learning rate 0.03).
5. **LSTM**: a small recurrent network that reads the 5 variables as a sequence (L4_u16: sequence length 4, units [16]).
6. **Stacked ensemble**: a linear meta-model combining Random Forest, XGBoost and LSTM predictions, fitted on validation-year predictions only (temporal holdout, not shuffled cross-validation).

Metrics: **MAE** (average absolute error, in score points), **RMSE** (penalises large misses), **R2** (share of variance explained; very high for every model here because countries differ a lot from each other, so compare errors with Persistence rather than reading R2 alone).

## 5. Results (chronological test: 133 countries, origin 2023, target 2024)

| Model | Validation RMSE | Test MAE | Test RMSE | Test R2 | Test RMSE vs Persistence (%) |
|---|---|---|---|---|---|
| Persistence | 0.1658 | 0.1037 | 0.1408 | 0.9848 | 0.0000 |
| Linear Regression | 0.1824 | 0.0938 | 0.1238 | 0.9882 | -12.1356 |
| Random Forest | 0.1973 | 0.1219 | 0.1771 | 0.9759 | 25.7487 |
| XGBoost | 0.1978 | 0.1352 | 0.1859 | 0.9735 | 31.9574 |
| LSTM | 0.2254 | 0.1592 | 0.2059 | 0.9675 | 46.1685 |
| Stacked Ensemble | 0.2048 | 0.1305 | 0.1760 | 0.9762 | 24.9586 |

Lowest test RMSE: **Linear Regression** (0.1238). Lowest validation RMSE among the five fitted models: **Linear Regression** (0.1824); Persistence has a validation RMSE of 0.1658. Validation numbers of tuned models are slightly optimistic; test numbers are the comparison to cite.

| Model | Test RMSE 95% interval (country bootstrap) |
|---|---|
| Persistence | 0.1212 to 0.1614 |
| Linear Regression | 0.1058 to 0.1434 |
| Random Forest | 0.1306 to 0.2374 |
| XGBoost | 0.1532 to 0.2210 |
| LSTM | 0.1778 to 0.2353 |
| Stacked Ensemble | 0.1468 to 0.2074 |

The intervals resample the 133 test countries for the saved predictions only. They describe uncertainty on this one test year, not the effect of retraining.

## 6. Experiment A: random split vs chronological evaluation

Random 80:20 splits (10 seeds) with the same fixed hyperparameters. About 83% of random test rows belong to countries that also have later years in training, and for 69% of them the answer itself appears as a training feature; a chronological split makes both 0%.

| Model | Chronological test RMSE | Random split mean RMSE | Random split std | RMSE / Persistence (chronological) | RMSE / Persistence (random) |
|---|---|---|---|---|---|
| Persistence | 0.1408 | 0.1905 | 0.0142 | 1.0000 | 1.0000 |
| Linear Regression | 0.1238 | 0.1832 | 0.0152 | 0.8786 | 0.9619 |
| Random Forest | 0.1771 | 0.2006 | 0.0162 | 1.2575 | 1.0532 |
| XGBoost | 0.1859 | 0.1972 | 0.0147 | 1.3196 | 1.0353 |

The two test sets contain different years, so compare the last two columns (RMSE relative to Persistence on the same rows) before drawing conclusions.

## 7. Experiment B: rolling-origin backtest

For each forecast year Y the models train on rows with target year before Y and are evaluated on target year Y (preprocessing refitted every year). Skipped years: [2018] (fewer than 180 training rows). LSTM and stacking were not repeated (time budget).

| Model | Forecast years | Mean MAE | Mean RMSE | Mean R2 | Std of RMSE across years | Years better than Persistence |
|---|---|---|---|---|---|---|
| Persistence | 6 | 0.1276 | 0.1850 | 0.9712 | 0.0392 | nan |
| Linear Regression | 6 | 0.1250 | 0.1820 | 0.9715 | 0.0501 | 4.0000 |
| Random Forest | 6 | 0.1436 | 0.2094 | 0.9635 | 0.0394 | 0.0000 |
| XGBoost | 6 | 0.1440 | 0.2021 | 0.9662 | 0.0324 | 0.0000 |

## 8. SHAP explanation (Random Forest, chosen on validation error)

| Variable (all years combined) | Sum of mean abs SHAP | Share of total (%) |
|---|---|---|
| HappinessScore | 0.9443 | 97.7883 |
| InternetAccessPct_WB | 0.0105 | 1.0827 |
| HealthExpenditurePctGDP_WB | 0.0072 | 0.7456 |
| log_GDPPerCapita_WB | 0.0025 | 0.2559 |
| UnemploymentRate_WB | 0.0012 | 0.1275 |

SHAP values describe what the model uses to forecast. They are **not causal effects**.

## 9. Repository structure

- `data/raw/`: downloaded source data (original and enriched)
- `data/processed_v2/`: calendar-aligned supervised table, splits, imputed/scaled versions, LSTM sequences, data dictionary
- `data/processed/`: the earlier row-shift pipeline output (kept for provenance)
- `models/`: saved models and preprocessing artifacts
- `results/tables/`: all result tables and reports
- `results/predictions/`: saved predictions for every model and experiment
- `results/figures/`: all figures (final_fig1..7 are the summary set)
- `src/`: shared helper code used by the notebook
- `notebooks/`: notebooks
- `ENVIRONMENT.md`: Python and package versions used
- `requirements.txt`: pinned packages

## 10. Install

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 11. Notebook execution order

The pipeline is a notebook (`notebooks/CountryWellness_pipeline_clean.ipynb` (outputs removed; run its cells in order in Google Colab)); each cell begins with `# CELL n`. Run in order:

| Cells | Purpose |
|---|---|
| 1-13 | audit of the original data, candidate feature vetting, coverage and multicollinearity checks |
| 14-16 | enriched dataset, validation, Git security cleanup |
| 17-21 | temporal alignment audit, strategy decision, missing-value policy, supervised table, validation |
| 22-26 | chronological split, train-only imputation and scaling, artifacts, data dictionary, Phase 1 audit |
| 27-32 | utilities, Persistence, Linear Regression, Random Forest, XGBoost, baseline checks |
| 33-38 | LSTM sequences, training, validation-based selection, one test evaluation |
| 39-42 | stacking, six-model comparison, Phase 2 validation |
| 43-54 | random-split experiment, rolling-origin backtest, SHAP, error analysis, figures, validation |
| 57-62 | environment audit, security cleanup, README, artifact audit, final commit, demo script |

Cells 55-56 are not used.

## 12. Reproduce

1. Open the notebook in Google Colab with Google Drive mounted at `/content/drive/MyDrive/CountryWellness/`.
2. Clone this repository into that folder and check out branch `modeling-phase`.
3. Run the cells in the order above. Cells that need a GitHub token ask for it with a hidden prompt; nothing is stored.
4. All experiments use random seed 42 and run on CPU in a few minutes.

## 13. Limitations

- The chronological test is a single origin year (2023 to 2024); differences between models may be within noise (see the bootstrap intervals).
- Strict calendar alignment reduces the training set to 842 rows; the LSTM has little data.
- Validation numbers of tuned models are optimistic; early stopping of the LSTM used validation rows; the stacking meta-model has only 133 training rows and highly correlated inputs.
- The random-split experiment compares evaluation protocols and uses different test rows than the chronological test.
- Hong Kong and Puerto Rico have no health-expenditure data (flagged, median-imputed).
- Feature importance, SHAP values and model weights describe prediction, not causation.

## 14. Credits

The original data-merging and preprocessing notebook is by Humaira201 (upstream repository `Humaira201/Country-Wellness-Score-Prediction`); the temporal redesign, models and experiments were added on top of it. Data: Our World in Data (World Happiness Report, World Bank, WHO).
