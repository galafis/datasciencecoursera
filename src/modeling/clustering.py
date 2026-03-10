"""
Clustering Module

Provides implementations of K-Means and DBSCAN clustering algorithms
using numpy, without relying on scikit-learn.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class KMeans:
    """
    K-Means clustering algorithm.

    Uses Lloyd's algorithm with random initialization and supports
    multiple restarts for better convergence.
    """

    def __init__(self, n_clusters: int = 3, max_iterations: int = 300,
                 tolerance: float = 1e-4, n_init: int = 10, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_init = n_init
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.inertia: float = float("inf")
        self._fitted = False

    def _init_centroids(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Initialize centroids by randomly selecting k data points."""
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid."""
        distances = np.sqrt(((X[:, np.newaxis] - centroids[np.newaxis, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids as the mean of assigned points."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
        return centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute the sum of squared distances to the nearest centroid."""
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return float(inertia)

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit the K-Means model.

        Args:
            X: Data matrix of shape (n_samples, n_features).

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] < self.n_clusters:
            raise ValueError("Number of samples must be >= number of clusters.")

        rng = np.random.RandomState(self.random_state)
        best_inertia = float("inf")
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            centroids = self._init_centroids(X, rng)

            for _ in range(self.max_iterations):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels)

                shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
                centroids = new_centroids

                if shift < self.tolerance:
                    break

            inertia = self._compute_inertia(X, labels, centroids)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia = best_inertia
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            X: Data matrix of shape (n_samples, n_features).

        Returns:
            Cluster labels as 1D array.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self._assign_clusters(X, self.centroids)

    def get_params(self) -> Dict[str, object]:
        """Return model parameters and results."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return {
            "n_clusters": self.n_clusters,
            "centroids": self.centroids.tolist(),
            "inertia": self.inertia,
        }


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Identifies clusters of varying shapes based on density, and marks
    low-density points as noise.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels: Optional[np.ndarray] = None
        self._fitted = False

    def _region_query(self, X: np.ndarray, point_idx: int) -> List[int]:
        """Find all points within eps distance of the given point."""
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return list(np.where(distances <= self.eps)[0])

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Fit the DBSCAN model.

        Args:
            X: Data matrix of shape (n_samples, n_features).

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                labels[i] = cluster_id
                seed_set = list(neighbors)
                seed_set.remove(i)

                j = 0
                while j < len(seed_set):
                    q = seed_set[j]
                    if not visited[q]:
                        visited[q] = True
                        q_neighbors = self._region_query(X, q)
                        if len(q_neighbors) >= self.min_samples:
                            for n in q_neighbors:
                                if n not in seed_set:
                                    seed_set.append(n)
                    if labels[q] == -1:
                        labels[q] = cluster_id
                    j += 1

                cluster_id += 1

        self.labels = labels
        self._fitted = True
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(X)
        return self.labels

    def get_params(self) -> Dict[str, object]:
        """Return model parameters and results."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = int(np.sum(self.labels == -1))

        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
        }
