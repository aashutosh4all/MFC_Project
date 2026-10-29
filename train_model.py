from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

TARGET = "Power_density"
GROUP = "Study_ID"
NUMERIC_FEATURES = [
    "HRT", "COD", "Temp", "Light_intensity", "Light_cycle", "Flowrate", "Resis", "M_DO"
]
CATEGORICAL_FEATURES = [
    "Reactor type", "Carbon source", "Light_source", "Anode", "Membrane",
    "Anolyte", "Anode_Inoc", "Catholyte", "Cathode_Inoc", "Algae"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}\nPut processed_v1.xlsx beside this script or pass --data PATH.")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Dataset must be an .xlsx, .xls, or .csv file.")
    df.columns = [str(c).strip() for c in df.columns]
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.loc[df[TARGET].notna() & (df[TARGET] >= 0)].copy()
    if df.empty:
        raise ValueError("No rows with a valid, non-negative Power_density target were found.")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        # Preserve true missing values for the imputer; normalize non-missing values to strings.
        df[col] = df[col].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="processed_v1.xlsx", help="Input Excel/CSV dataset")
    parser.add_argument("--output", default="mfc_power_density_model.pkl", help="Output model filename")
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    data_path = Path(args.data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    df = read_dataset(data_path)
    X, y = df[FEATURES].copy(), df[TARGET].astype(float).to_numpy()
    groups = df[GROUP].astype(str).to_numpy() if GROUP in df.columns else None

    print(f"Rows used: {len(df)}")
    print(f"Unique Study_ID groups: {pd.Series(groups).nunique() if groups is not None else 'unavailable'}")
    print(f"Features: {len(FEATURES)} (8 numeric, 10 categorical)")
    print("Target summary:")
    print(pd.Series(y, name=TARGET).describe().to_string())

    # One random-row holdout, retained for comparability with the earlier project results.
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=42)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    candidates = {
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=5, min_samples_split=6,
            min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=2,
            min_samples_leaf=3, loss="huber", random_state=42
        ),
        "SVR": SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale"),
    }
    benchmark = {}
    for name, estimator in candidates.items():
        # Fit on log1p(target); invert predictions to the original target units.
        pipe = make_pipeline(estimator)
        pipe.fit(X_train, np.log1p(y_train))
        pred = np.maximum(0.0, np.expm1(pipe.predict(X_test)))
        benchmark[name] = metrics(y_test, pred)
    print("\nRandom holdout results (20% test; metrics on original target scale):")
    print(pd.DataFrame(benchmark).T.round(4).to_string())

    # Grouped CV prevents rows from the same source study appearing in both train and test.
    if groups is not None and len(np.unique(groups)) >= 2:
        n_splits = min(5, len(np.unique(groups)))
        cv = GroupKFold(n_splits=n_splits)
        grouped_results = {}
        for name, estimator in candidates.items():
            oof = np.full(len(df), np.nan, dtype=float)
            for tr, te in cv.split(X, y, groups):
                fold_pipe = make_pipeline(estimator)
                fold_pipe.fit(X.iloc[tr], np.log1p(y[tr]))
                oof[te] = np.maximum(0.0, np.expm1(fold_pipe.predict(X.iloc[te])))
            grouped_results[name] = metrics(y, oof)
        print(f"\n{n_splits}-fold GroupKFold results (original target scale):")
        print(pd.DataFrame(grouped_results).T.round(4).to_string())
    else:
        print("\nGrouped CV skipped: Study_ID is absent or has fewer than two groups.")

    # Fit final tuned RF pipeline on all target-valid rows, as in the recorded project workflow.
    final_estimator = RandomForestRegressor(
        n_estimators=500, max_depth=5, min_samples_split=6,
        min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1
    )
    final_pipe = make_pipeline(final_estimator)
    final_pipe.fit(X, np.log1p(y))
    package = {
        "pipeline": final_pipe,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "target_transform": "log1p; inference uses expm1",
        "training_rows": int(len(df)),
        "study_groups": int(pd.Series(groups).nunique()) if groups is not None else None,
        "random_holdout_metrics": benchmark,
        "grouped_cv_metrics": grouped_results if groups is not None and len(np.unique(groups)) >= 2 else None,
        "note": "Research baseline. Random holdout performance is not evidence of generalisation to unseen studies.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, output_path)
    reloaded = joblib.load(output_path)
    sample = X.iloc[:min(5, len(X))]
    p1 = np.expm1(final_pipe.predict(sample))
    p2 = np.expm1(reloaded["pipeline"].predict(sample))
    max_diff = float(np.max(np.abs(p1 - p2)))
    print(f"\nSaved model package: {output_path}")
    print(f"Reload check maximum prediction difference: {max_diff:.3e}")
    with open(output_path.with_suffix(".metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"random_holdout": benchmark, "grouped_cv": package["grouped_cv_metrics"]}, f, indent=2)
    print(f"Saved metrics: {output_path.with_suffix('.metrics.json')}")

if __name__ == "__main__":
    main()
