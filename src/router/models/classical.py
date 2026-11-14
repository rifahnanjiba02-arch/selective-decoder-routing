"""Classical rescue routers (Week 2, Task 3).

Predict ``y_router`` (1 = PyMatching wrong AND correlated matching correct)
from the Task 1 feature matrix. Rules enforced here:

* Preprocessing gets an explicit allow-list of columns and uses
  ``remainder="drop"``; nothing unexpected is passed through.
* Forbidden fields (labels, decoder outputs, correctness flags, target,
  experiment id) are rejected *before* fitting.
* Imputer / scaler / one-hot encoder are fitted on train only (inside the
  Pipeline, so ``predict`` never refits them).
* Each model returns one rescue probability in [0, 1] per row. These scores
  are not calibrated unless calibration is separately checked.
* No data loading happens here, so the test split is never touched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_NAMES = ("dummy", "logistic_regression", "random_forest", "xgboost")

# Must never be a column of X.
FORBIDDEN_COLUMNS = frozenset(
    {
        "label",
        "pymatching",
        "correlated_matching",
        "pymatching_correct",
        "correlated_matching_correct",
        "correlated_correct",
        "decoder_correct",
        "y_router",
        "target",
        "experiment",
    }
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def load_model_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and sanity-check ``configs/models.yaml``."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, Mapping) or "models" not in cfg:
        raise ValueError(f"{path}: missing top-level 'models' section")
    missing = [m for m in MODEL_NAMES if m not in cfg["models"]]
    if missing:
        raise ValueError(f"{path}: missing model configs: {missing}")
    if "seed" not in cfg:
        raise ValueError(f"{path}: missing 'seed'")
    cfg.setdefault("categorical_columns", ["basis"])
    return dict(cfg)


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------
def check_allowed_columns(allowed_columns: Sequence[str]) -> list[str]:
    allowed = list(allowed_columns)
    if not allowed:
        raise ValueError("allowed_columns is empty")
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed_columns contains duplicates")
    bad = sorted(set(allowed) & FORBIDDEN_COLUMNS)
    if bad:
        raise ValueError(f"forbidden fields in allowed_columns: {bad}")
    return allowed


def validate_features(X: pd.DataFrame, allowed_columns: Sequence[str]) -> pd.DataFrame:
    """Reject forbidden columns, require all allowed ones, return X[allowed].

    Other unexpected columns are dropped here (and again by
    ``remainder="drop"``), so they can never reach a model.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    allowed = check_allowed_columns(allowed_columns)
    present_forbidden = sorted(set(X.columns) & FORBIDDEN_COLUMNS)
    if present_forbidden:
        raise ValueError(f"forbidden columns present in X: {present_forbidden}")
    missing = [c for c in allowed if c not in X.columns]
    if missing:
        raise ValueError(f"X is missing allowed columns: {missing}")
    return X.loc[:, allowed]


def validate_target(y: Iterable, n_rows: int, require_both_classes: bool = True) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if len(arr) != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {len(arr)}")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("y must contain only 0/1")
    arr = arr.astype(int)
    if require_both_classes and len(np.unique(arr)) < 2:
        raise ValueError("training target must contain both classes")
    return arr


# --------------------------------------------------------------------------
# pipeline construction
# --------------------------------------------------------------------------
def build_preprocessor(
    allowed_columns: Sequence[str],
    categorical_columns: Sequence[str] = ("basis",),
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Explicit-allow-list preprocessor with ``remainder='drop'``."""
    allowed = check_allowed_columns(allowed_columns)
    cats = [c for c in categorical_columns if c in allowed]
    nums = [c for c in allowed if c not in cats]
    if "basis" in allowed and "basis" not in cats:
        raise ValueError("'basis' must be treated as a categorical column")

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))

    transformers = []
    if nums:
        transformers.append(("num", Pipeline(num_steps), nums))
    if cats:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cats,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def _make_estimator(name: str, params: Mapping[str, Any], seed: int, y_train: np.ndarray):
    p = dict(params)
    p.pop("scale_numeric", None)
    if name == "dummy":
        return DummyClassifier(strategy=p.get("strategy", "prior"), random_state=seed)
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **p)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **p)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:  # pragma: no cover
            raise ImportError("xgboost is not installed (pip install xgboost)") from e
        if p.get("scale_pos_weight") == "auto":
            n_pos = int((y_train == 1).sum())
            p["scale_pos_weight"] = float((y_train == 0).sum() / max(n_pos, 1))
        return XGBClassifier(
            random_state=seed, objective="binary:logistic", eval_metric="logloss", **p
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


def build_pipeline(
    name: str,
    allowed_columns: Sequence[str],
    y_train: Sequence[int],
    config: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Preprocessor + estimator for one candidate (unfitted)."""
    cfg = dict(config) if config is not None else load_model_config()
    if name not in cfg["models"]:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    params = cfg["models"][name]
    pre = build_preprocessor(
        allowed_columns,
        categorical_columns=cfg.get("categorical_columns", ["basis"]),
        scale_numeric=bool(params.get("scale_numeric", False)),
    )
    est = _make_estimator(name, params, int(cfg["seed"]), np.asarray(y_train).astype(int))
    return Pipeline([("prep", pre), ("model", est)])


# --------------------------------------------------------------------------
# fit / predict
# --------------------------------------------------------------------------
def fit_router(
    name: str,
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    allowed_columns: Sequence[str],
    config: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Fit one candidate on TRAIN ONLY and return the fitted pipeline."""
    X = validate_features(X_train, allowed_columns)
    y = validate_target(y_train, len(X))
    pipe = build_pipeline(name, allowed_columns, y, config)
    pipe.fit(X, y)
    pipe.allowed_columns_ = list(allowed_columns)  # recorded for predict-time checks
    return pipe


def predict_rescue_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """One rescue probability in [0, 1] per row (class 1). Not calibrated."""
    allowed = getattr(pipe, "allowed_columns_", None)
    if allowed is None:
        raise ValueError("pipeline was not fitted via fit_router()")
    Xv = validate_features(X, allowed)
    proba = pipe.predict_proba(Xv)
    classes = list(pipe.classes_)
    if 1 not in classes:
        raise ValueError("fitted model has no positive class")
    scores = np.asarray(proba[:, classes.index(1)], dtype=float)
    if scores.shape != (len(Xv),):
        raise ValueError("score shape does not match number of rows")
    if not np.isfinite(scores).all() or scores.min() < 0.0 or scores.max() > 1.0:
        raise ValueError("scores must be finite and in [0, 1]")
    return scores


def fit_all_routers(
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    allowed_columns: Sequence[str],
    config: Mapping[str, Any] | None = None,
    names: Sequence[str] = MODEL_NAMES,
) -> dict[str, Pipeline]:
    """Fit every candidate on the same train matrix and target."""
    cfg = dict(config) if config is not None else load_model_config()
    return {n: fit_router(n, X_train, y_train, allowed_columns, cfg) for n in names}


def predict_all_routers(models: Mapping[str, Pipeline], X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Aligned score arrays (same rows, same order) for every fitted model."""
    return {n: predict_rescue_proba(m, X) for n, m in models.items()}


def model_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Settings dict to print/record in notebook 02 and the findings note."""
    cfg = dict(config) if config is not None else load_model_config()
    return {
        "seed": cfg["seed"],
        "categorical_columns": cfg.get("categorical_columns", ["basis"]),
        "models": {k: dict(v) for k, v in cfg["models"].items()},
    }
