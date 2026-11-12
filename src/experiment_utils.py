
"""Shared code for Phase 3 experiments. Every function that fits preprocessing fits it on the TRAINING rows it is given only."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

SEED = 42
MODELS_CHEAP = ["Persistence", "Linear Regression", "Random Forest", "XGBoost"]
COLORS = {"Persistence": "#7f7f7f", "Linear Regression": "#1f77b4", "Random Forest": "#2ca02c", "XGBoost": "#ff7f0e",
          "LSTM": "#9467bd", "Stacked Ensemble": "#d62728"}
RF_KEYS = ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "n_jobs", "random_state")
XGB_KEYS = ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "tree_method", "n_jobs", "random_state", "verbosity")


def style():
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
                         "legend.fontsize": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": True})


def load_fixed_params(P):
    """Hyperparameters chosen in Phase 2 on VALIDATION. Never changed in Phase 3."""
    import json, os
    rf = json.load(open(os.path.join(P["models"], "random_forest_config.json")))
    xgb = json.load(open(os.path.join(P["models"], "xgboost_config.json")))
    return {"rf": {k: rf[k] for k in RF_KEYS}, "xgb": {k: xgb[k] for k in XGB_KEYS}}


def fit_predict(model_name, train_df, test_df, features, flags, target, params):
    """Fit one model on train_df ONLY (imputer, scaler and model) and predict test_df. Returns a numpy array."""
    if model_name == "Persistence":
        return test_df["HappinessScore"].to_numpy(float)                  # next year = this year, nothing is fitted
    cont = [c for c in features if c not in flags]
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(train_df[features])      # medians from TRAIN only
    Xtr = pd.DataFrame(imp.transform(train_df[features]), columns=features, index=train_df.index)
    Xte = pd.DataFrame(imp.transform(test_df[features]), columns=features, index=test_df.index)
    y = train_df[target].to_numpy(float)
    if model_name == "Linear Regression":
        sc = StandardScaler().fit(Xtr[cont])                                                       # mean/std from TRAIN only
        Xtr, Xte = Xtr.copy(), Xte.copy()
        Xtr[cont], Xte[cont] = sc.transform(Xtr[cont]), sc.transform(Xte[cont])
        return LinearRegression().fit(Xtr, y).predict(Xte)
    if model_name == "Random Forest":
        return RandomForestRegressor(**params["rf"]).fit(Xtr, y).predict(Xte)
    if model_name == "XGBoost":
        return XGBRegressor(**params["xgb"]).fit(Xtr, y).predict(Xte)
    raise ValueError(f"unknown model {model_name}")


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=SEED, base_pred=None):
    """Percentile 95% CI by resampling countries (rows) with replacement. FIXED predictions: no retraining is simulated."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    mae, rmse, d_rmse = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        e = y_pred[i] - y_true[i]
        mae.append(np.abs(e).mean()); rmse.append(np.sqrt((e ** 2).mean()))
        if base_pred is not None:
            eb = np.asarray(base_pred, float)[i] - y_true[i]
            d_rmse.append(np.sqrt((e ** 2).mean()) - np.sqrt((eb ** 2).mean()))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    out = {"MAE_lo": q(mae)[0], "MAE_hi": q(mae)[1], "RMSE_lo": q(rmse)[0], "RMSE_hi": q(rmse)[1]}
    if base_pred is not None:
        out.update({"dRMSE_vs_persistence_lo": q(d_rmse)[0], "dRMSE_vs_persistence_hi": q(d_rmse)[1],
                    "share_boot_better_than_persistence": float(np.mean(np.array(d_rmse) < 0))})
    return out
