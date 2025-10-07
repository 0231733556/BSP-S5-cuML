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

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score


@dataclass
class ClassifierWrapper:
    """A small wrapper around scikit-learn estimators with an optional scaler.

    Fields:
        estimator_name: one of 'random_forest', 'logistic', 'svc' (default 'random_forest')
        use_scaler: whether to add StandardScaler in the pipeline
        random_state: optional random seed for reproducibility
    """

    estimator_name: str = 'random_forest'
    use_scaler: bool = True
    random_state: Optional[int] = 42

    def __post_init__(self):
        self.estimator = self._make_estimator(self.estimator_name)
        steps = []
        if self.use_scaler:
            steps.append(('scaler', StandardScaler()))
        steps.append(('estimator', self.estimator))
        self.pipeline: Pipeline = Pipeline(steps)

    def _make_estimator(self, name: str):
        name = name.lower()
        if name in ('random_forest', 'rf'):
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        if name in ('logistic', 'logistic_regression', 'lr'):
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
        if name in ('svc', 'svm'):
            return SVC(probability=True, random_state=self.random_state)
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
        self.algorithm_params = self.algorithm_params or {}
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


