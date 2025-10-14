"""General scikit-learn classifier utilities.

Provides a simple ClassifierWrapper that builds a pipeline with an optional
StandardScaler and any scikit-learn estimator. Includes train, predict,
save and load helpers and a small example runner that uses the Iris dataset
for a quick smoke test.

Usage:
    from models import ClassifierWrapper
    clf = ClassifierWrapper(estimator_name='random_forest')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

This file is intentionally self-contained and uses only numpy/pandas and
scikit-learn.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Optional
import logging
import numpy as np
import pandas as pd
import time
import statistics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
import copy
LOG = logging.getLogger(__name__)

def set_log_level(level):
    logging.basicConfig(level=level)
    LOG.setLevel(level)
    
# Module-level defaults that can be changed by calling `set_defaults`
DEFAULTS = {
    "classifier": {
        "estimator_name": "random_forest",
        "use_scaler": True,
        "random_state": 42,
        "n_estimators": 100,
        "max_iter": 1000,
        "probability": True,
    },
    "clustering": {
        "algorithm_name": "kmeans",
        "n_clusters": 3,
        "use_scaler": True,
        "random_state": 42,
        "algorithm_params": None,
    },
}


def set_defaults(classifier: dict | None = None, clustering: dict | None = None) -> None:
    """Update module-level defaults.

    Examples:
        set_defaults(classifier={'n_estimators': 200})
        set_defaults(clustering={'n_clusters': 10})
    """
    if classifier:
        DEFAULTS["classifier"].update(classifier)
    if clustering:
        DEFAULTS["clustering"].update(clustering)


def get_defaults() -> dict:
    """Return a deep copy of the current defaults."""
    return copy.deepcopy(DEFAULTS)


@dataclass
class ClassifierWrapper:
    """A small wrapper around scikit-learn estimators with an optional scaler.

    Fields:
        estimator_name: one of 'random_forest', 'logistic', 'svc' (default 'random_forest')
        use_scaler: whether to add StandardScaler in the pipeline
        random_state: optional random seed for reproducibility
    """

    estimator_name: None | str = None
    use_scaler: bool = True
    random_state: Optional[int] = 42
    n_estimators: int = 100
    max_iter: int = 1000
    probability: bool = True

    def __post_init__(self):
        # Fill missing/None fields from module DEFAULTS
        d = DEFAULTS["classifier"]
        if self.estimator_name is None:
            self.estimator_name = d["estimator_name"]
        if self.use_scaler is None:
            self.use_scaler = d["use_scaler"]
        if self.random_state is None:
            self.random_state = d["random_state"]
        if getattr(self, "n_estimators", None) is None:
            self.n_estimators = d["n_estimators"]
        if getattr(self, "max_iter", None) is None:
            self.max_iter = d["max_iter"]
        if getattr(self, "probability", None) is None:
            self.probability = d["probability"]

        self.estimator = self._make_estimator(self.estimator_name)
        steps = []
        if self.use_scaler:
            steps.append(('scaler', StandardScaler()))
        steps.append(('estimator', self.estimator))
        self.pipeline: Pipeline = Pipeline(steps)

    def _make_estimator(self, name: str):
        name = name.lower()
        if name in ('random_forest', 'rf','randomforestclassifier','RandomForestClassifier'):
            return RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        if name in ('logistic', 'logistic_regression', 'lr'):
            return LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        if name in ('svc', 'svm'):
            return SVC(probability=self.probability, random_state=self.random_state)
        raise ValueError(f"unknown estimator_name: {name}")

    def fit(self, X: Any, y: Any) -> 'ClassifierWrapper':
        """Fit the internal pipeline.

        Accepts numpy arrays, pandas DataFrame/Series, or list-likes.
        """
        X_arr, y_arr = self._to_arrays(X, y)
        self.pipeline.fit(X_arr, y_arr)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = self._to_array(X)
        return self.pipeline.predict(X_arr)

    def predict_proba(self, X: Any) -> np.ndarray:
        X_arr = self._to_array(X)
        # Some estimators may not support predict_proba; raise if missing
        if not hasattr(self.pipeline, 'predict_proba') and not hasattr(self.pipeline[-1], 'predict_proba'):
            raise AttributeError('Underlying estimator does not support predict_proba')
        return self.pipeline.predict_proba(X_arr)

    def save(self, path: str) -> None:
        """Save the fitted pipeline to disk using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load(self, path: str) -> None:
        """Load a pipeline from disk (replaces the current pipeline)."""
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)

    def _to_array(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.values
        if isinstance(X, np.ndarray):
            return X
        return np.array(X)

    def _to_arrays(self, X: Any, y: Any):
        return self._to_array(X), self._to_array(y)

@dataclass
class ClusteringWrapper:
    """A wrapper around clustering algorithms with an optional scaler.

    Fields:
        algorithm_name: one of 'kmeans', 'agglomerative', 'dbscan' (default 'kmeans')
        n_clusters: number of clusters (ignored by DBSCAN)
        use_scaler: whether to add StandardScaler in the pipeline
        random_state: optional random seed for reproducibility
        algorithm_params: dict of extra estimator params
    """

    algorithm_name: str = "kmeans"
    n_clusters: int = 3
    use_scaler: bool = True
    random_state: Optional[int] = 42
    algorithm_params: Optional[dict] = None

    def __post_init__(self):
        # Fill missing/None fields from module DEFAULTS
        d = DEFAULTS["clustering"]
        if self.algorithm_name is None:
            self.algorithm_name = d["algorithm_name"]
        if self.n_clusters is None:
            self.n_clusters = d["n_clusters"]
        if self.use_scaler is None:
            self.use_scaler = d["use_scaler"]
        if self.random_state is None:
            self.random_state = d["random_state"]
        if self.algorithm_params is None:
            self.algorithm_params = d["algorithm_params"] or {}

        self.estimator = self._make_estimator(self.algorithm_name)
        steps = []
        if self.use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("estimator", self.estimator))
        self.pipeline: Pipeline = Pipeline(steps)

    def _make_estimator(self, name: str):
        name = name.lower()
        if name in ("kmeans", "k-means", "km"):
            return KMeans(n_clusters=self.n_clusters, random_state=self.random_state, **self.algorithm_params)
        if name in ("agglomerative", "agglomerative_clustering", "agg"):
            return AgglomerativeClustering(n_clusters=self.n_clusters, **self.algorithm_params)
        if name in ("dbscan",):
            return DBSCAN(**self.algorithm_params)
        raise ValueError(f"unknown algorithm_name: {name}")

    def fit(self, X: Any) -> "ClusteringWrapper":
        X_arr = self._to_array(X)
        # Most clustering estimators implement fit; for ones that only
        # implement fit_predict (e.g., DBSCAN) we call fit on the estimator.
        try:
            self.pipeline.fit(X_arr)
        except Exception:
            # Fallback: fit estimator directly
            self.pipeline.named_steps["estimator"].fit(X_arr)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_arr = self._to_array(X)
        # If estimator supports predict (KMeans), use pipeline.predict
        if hasattr(self.pipeline, "predict"):
            return self.pipeline.predict(X_arr)
        # Otherwise, if estimator supports fit_predict, use it (will refit)
        est = self.pipeline.named_steps["estimator"]
        if hasattr(est, "fit_predict"):
            return est.fit_predict(X_arr)
        raise AttributeError("Underlying estimator does not support predict or fit_predict")

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)

    def _to_array(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.values
        if isinstance(X, np.ndarray):
            return X
        return np.array(X)

    def _to_arrays(self, X: Any, y: Any):
        return self._to_array(X), self._to_array(y)

def train_model(
    model,
    X,
    y=None,
    *,
    X_val=None,
    y_val=None,
    trials: int = 1,
    warmups: int = 0,
    metric=None,
    random_state: Optional[int] = None,
):
    """Generic training helper.

    Fits `model` on (X, y) and returns timing statistics. Optionally evaluates
    on validation data using `metric` (callable taking (y_true, y_pred)).

    Parameters
    - model: object with a .fit(X, y) method. May be a pipeline or wrapper.
    - X, y: training data. For unsupervised models y can be None.
    - X_val, y_val: optional validation data for scoring after fit.
    - trials: number of timed trials to run (re-fit each trial).
    - warmups: number of warmup runs before timing.
    - metric: optional callable(y_true, y_pred) to compute a validation score.
    - random_state: optional seed to set on the model if it has that attribute.

    Returns a dict with keys: 'times' (list), 'median', 'mean', 'std', and
    optionally 'val_score_median' and 'val_scores'.
    """

    def _set_seed(m, seed):
        if seed is None:
            return
        # Prefer attribute on estimator, fall back to setting on named step
        try:
            if hasattr(m, "random_state"):
                setattr(m, "random_state", seed)
        except Exception:
            pass

    # Warmups
    for _ in range(warmups):
        _set_seed(model, random_state)
        if y is None:
            model.fit(X)
        else:
            model.fit(X, y)

    times = []
    val_scores = []
    for _ in range(trials):
        _set_seed(model, random_state)
        t0 = time.perf_counter()
        if y is None:
            model.fit(X)
        else:
            model.fit(X, y)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        if X_val is not None and y_val is not None and metric is not None:
            preds = None
            try:
                preds = model.predict(X_val)
            except Exception:
                est = getattr(model, "estimator", model)
                if hasattr(est, "fit_predict"):
                    try:
                        preds = est.fit_predict(X_val)
                    except Exception:
                        preds = None
            if preds is not None:
                try:
                    val_scores.append(metric(y_val, preds))
                except Exception:
                    val_scores.append(None)

    result = {
        "times": times,
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "std": statistics.pstdev(times) if len(times) > 1 else 0.0,
    }
    if val_scores:
        filtered = [s for s in val_scores if s is not None]
        if filtered:
            result["val_score_median"] = statistics.median(filtered)
            result["val_scores"] = val_scores
    return result

def use_accel(val: bool):
    if val:
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                LOG.debug("Loading cuML accel extension")
                ipython.run_line_magic("load_ext", "cuml.accel")
                
        except Exception:
            pass 
    else:
        LOG.debug("Skipping cuML accel extension load failure")