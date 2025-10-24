from sklearn.datasets import make_classification, make_blobs
from scipy.sparse import random as sparse_random
import numpy as np

def generate_classification_data(
    n_samples=100_000,
    n_features=50,
    n_informative=10,
    n_classes=2,
    sparsity=0.0,
    random_state=42,
):
    """
    Generate synthetic classification data.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    n_informative : int
        Number of informative features.
    n_classes : int
        Number of classes.
    sparsity : float
        Fraction of features set to zero (0 = dense, 0.9 = 90% zeros).
    random_state : int
        Random seed.

    Returns
    -------
    (X, y) : tuple of np.ndarray
        Feature matrix and target vector.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=random_state,
    )

    if sparsity > 0:
        mask = np.random.rand(*X.shape) > sparsity
        X = X * mask

    return X, y

def generate_clustering_data(
    n_samples=100_000,
    n_features=50,
    centers=10,
    cluster_std=1.0,
    random_state=42,
):
    """
    Generate synthetic clustering data.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    centers : int
        Number of cluster centers.
    cluster_std : float
        Standard deviation of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    (X, y) : tuple of np.ndarray
        Feature matrix and cluster labels.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
        return_centers=False,
    )
    return X, y