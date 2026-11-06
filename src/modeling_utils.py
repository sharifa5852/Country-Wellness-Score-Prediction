
"""Shared helpers for Phase 2 (loading, metrics, prediction-file validation). No test labels are used for fitting anywhere here."""
import os, json, shutil, datetime, platform
import importlib.metadata as md
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
KEYS = ["Code", "Year", "TargetYear"]
SPLITS = ("train", "val", "test")


def make_paths(root):
    P = {"root": root, "v2": os.path.join(root, "data", "processed_v2"), "prep": os.path.join(root, "models", "preprocessing"),
         "models": os.path.join(root, "models"), "pred": os.path.join(root, "results", "predictions"),
         "tables": os.path.join(root, "results", "tables"), "figures": os.path.join(root, "results", "figures"),
         "backups": os.path.join(root, "results", "_backups"), "raw": os.path.join(root, "data", "raw")}
    for k in ("models", "pred", "tables", "figures", "backups"):
        os.makedirs(P[k], exist_ok=True)
    os.makedirs(os.path.join(P["models"], "lstm"), exist_ok=True)
    return P


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    assert y_true.shape == y_pred.shape and y_true.ndim == 1, f"shape mismatch {y_true.shape} vs {y_pred.shape}"
    assert np.isfinite(y_true).all() and np.isfinite(y_pred).all(), "NaN/inf in metric inputs"
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)), "n": int(len(y_true))}


def _backup(path, P):
    if os.path.exists(path):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(path, os.path.join(P["backups"], f"{stamp}_{os.path.basename(path)}"))


def save_csv(df, path, P):
    """Write a CSV. If a different version exists, back it up first. Identical content is left alone."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        if old.shape == df.shape and list(old.columns) == list(df.columns):
            same = True
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]) and pd.api.types.is_numeric_dtype(old[c]):
                    same &= bool(np.allclose(old[c].to_numpy(float), df[c].to_numpy(float), atol=1e-9, equal_nan=True))
                else:
                    same &= bool((old[c].astype(str).to_numpy() == df[c].astype(str).to_numpy()).all())
            if same:
                return path
        _backup(path, P)
    df.to_csv(path, index=False)
    return path


def save_json(obj, path, P):
    if os.path.exists(path):
        if json.load(open(path)) == json.loads(json.dumps(obj, default=str)):
            return path
        _backup(path, P)
    json.dump(obj, open(path, "w"), indent=2, default=str)
    return path


def save_joblib(obj, path, P):
    import joblib
    _backup(path, P)
    joblib.dump(obj, path)
    return path


def load_phase1(P):
    """Load spec, imputed (unscaled), scaled, raw splits and the 'truth' tables."""
    spec = json.load(open(os.path.join(P["v2"], "feature_spec.json")))
    F, T = spec["feature_cols"], spec["target_col"]
    d = {"spec": spec, "features": F, "target": T, "raw": {}, "imp": {}, "sc": {}, "truth": {}}
    for s in SPLITS:
        d["raw"][s] = pd.read_csv(os.path.join(P["v2"], f"{s}.csv"))
        d["imp"][s] = pd.read_csv(os.path.join(P["v2"], f"{s}_imputed.csv"))
        d["sc"][s] = pd.read_csv(os.path.join(P["v2"], f"{s}_scaled.csv"))
        t = d["imp"][s][["Country", "Code", "Year", "TargetYear"]].copy()
        t["y_true"] = d["imp"][s][T].to_numpy(float)
        t["persistence"] = d["imp"][s][spec["persistence_col"]].to_numpy(float)
        d["truth"][s] = t
    return d


def X(d, kind, split):
    """kind: 'imp' = imputed, unscaled (trees);  'sc' = imputed + scaled (linear)."""
    return d[kind][split][d["features"]]


def pred_frame(truth, y_pred, split, model):
    f = truth[["Country", "Code", "Year", "TargetYear", "y_true"]].copy().reset_index(drop=True)
    f.insert(4, "split", split)
    f.insert(5, "model", model)
    f["y_pred"] = np.asarray(y_pred, float)
    return f


def validate_pred_frame(pf, truth, split, label=""):
    """Prediction rows must match the real observations one-to-one by (Code, Year, TargetYear)."""
    tag = f"[{label} {split}]"
    assert pf["y_pred"].notna().all() and np.isfinite(pf["y_pred"].to_numpy(float)).all(), f"{tag} NaN/inf prediction"
    assert not pf.duplicated(KEYS).any(), f"{tag} duplicate prediction rows"
    m = pf[KEYS + ["y_true"]].merge(truth[KEYS + ["y_true"]], on=KEYS, how="outer", suffixes=("_p", "_t"), indicator=True)
    assert (m["_merge"] == "both").all(), f"{tag} rows do not match the real test/val observations ({(m['_merge'] != 'both').sum()} differ)"
    assert np.allclose(m["y_true_p"], m["y_true_t"], atol=1e-12), f"{tag} y_true differs from the real target"
    assert (pf["TargetYear"] == pf["Year"] + 1).all(), f"{tag} TargetYear != Year+1"
    return True


def split_metrics(pf, split):
    s = pf[pf["split"] == split]
    return metrics(s["y_true"], s["y_pred"])


def check_phase1(P, d):
    """Return a DataFrame of PASS/FAIL checks. Raises nothing; the caller decides."""
    import joblib
    rows = []
    add = lambda n, ok, det="": rows.append({"check": n, "result": "PASS" if ok else "FAIL", "detail": str(det)})
    F, T, spec = d["features"], d["target"], d["spec"]
    need = [f"{s}{k}.csv" for s in SPLITS for k in ("", "_imputed", "_scaled")] + ["feature_spec.json", "lstm_sequences.npz", "feature_metadata.json", "preprocessing_metadata.json"]
    add("all Phase 1 data files exist", all(os.path.exists(os.path.join(P["v2"], f)) for f in need))
    add("all 4 preprocessing artifacts exist", all(os.path.exists(os.path.join(P["prep"], f)) for f in
        ("tabular_imputer.joblib", "tabular_scaler.joblib", "lstm_imputer.joblib", "lstm_scaler.joblib")))
    add("shapes 842/133/133 x 18 features", [len(d["imp"][s]) for s in SPLITS] == [842, 133, 133] and len(F) == 18, [len(d["imp"][s]) for s in SPLITS])
    add("feature order equal in spec, imputed, scaled", all(list(d[k][s].columns[5:]) == F for k in ("imp", "sc") for s in SPLITS))
    add("target unscaled (equals raw split target)", all(np.allclose(d[k][s][T], d["raw"][s][T]) for k in ("imp", "sc") for s in SPLITS))
    add("target range looks like a 0-10 score", 0 < d["raw"]["train"][T].min() and d["raw"]["train"][T].max() < 10,
        (round(float(d["raw"]["train"][T].min()), 2), round(float(d["raw"]["train"][T].max()), 2)))
    add("no NaN/inf in model inputs", all(np.isfinite(d[k][s][F].to_numpy(float)).all() for k in ("imp", "sc") for s in SPLITS))
    add("feature/target separation (target not among features)", T not in F and "TargetYear" not in F and "has_lag3" not in F)
    add("test targets are all 2024, val targets 2023, train targets <= 2022",
        d["truth"]["test"].TargetYear.eq(2024).all() and d["truth"]["val"].TargetYear.eq(2023).all() and d["truth"]["train"].TargetYear.max() == 2022)
    keyset = lambda s: set(zip(d["truth"][s].Code, d["truth"][s].Year))
    add("train/val/test rows do not overlap", not (keyset("train") & keyset("val")) and not (keyset("train") & keyset("test")) and not (keyset("val") & keyset("test")))
    sc = joblib.load(os.path.join(P["prep"], "tabular_scaler.joblib"))
    add("scaler was fit on exactly the training rows (842)", int(sc.n_samples_seen_) == 842, int(sc.n_samples_seen_))
    return pd.DataFrame(rows)


def load_seq(P, L, splits=("train", "val", "test")):
    """Load LSTM arrays (n, L, 5), oldest -> newest. Only the requested splits are read from disk."""
    d_truth = {}
    out = {}
    if L == 3:
        z = np.load(os.path.join(P["v2"], "lstm_sequences.npz"))
        for s in splits:
            t = pd.read_csv(os.path.join(P["v2"], f"{s}_imputed.csv"))[["Country", "Code", "Year", "TargetYear"]]
            out[s] = {"X": z[f"X_{s}"].astype(np.float32), "y": z[f"y_{s}"].astype(np.float32), "keys": t.reset_index(drop=True), "row_idx": np.arange(len(t))}
    elif L == 4:
        z = np.load(os.path.join(P["v2"], "lstm_seq_len4.npz"), allow_pickle=False)
        for s in splits:
            t = pd.read_csv(os.path.join(P["v2"], f"{s}_imputed.csv"))[["Country", "Code", "Year", "TargetYear"]]
            idx = z[f"row_idx_{s}"]
            out[s] = {"X": z[f"X_{s}"].astype(np.float32), "y": z[f"y_{s}"].astype(np.float32), "keys": t.iloc[idx].reset_index(drop=True), "row_idx": idx}
    else:
        raise ValueError("L must be 3 or 4")
    for s in splits:
        o = out[s]
        assert o["X"].ndim == 3 and o["X"].shape[1] == L and o["X"].shape[2] == 5, f"bad shape {o['X'].shape}"
        assert len(o["X"]) == len(o["y"]) == len(o["keys"]) and np.isfinite(o["X"]).all()
    return out


def env_info():
    info = {"python": platform.python_version(), "platform": platform.platform(), "seed": SEED, "cpu_count": os.cpu_count()}
    for pkg in ("numpy", "pandas", "scikit-learn", "xgboost", "tensorflow", "keras", "joblib", "matplotlib"):
        try:
            info[pkg] = md.version(pkg)
        except Exception:
            info[pkg] = "not installed"
    info["gpu_tool_present"] = shutil.which("nvidia-smi") is not None
    return info
