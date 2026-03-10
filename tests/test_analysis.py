"""
Comprehensive tests for the data science portfolio.

Tests cover EDA, hypothesis testing, regression, clustering,
and preprocessing modules.
"""

import numpy as np
import pytest

from src.analysis.eda import (
    descriptive_stats, column_stats, frequency_distribution,
    correlation_matrix, detect_outliers_iqr, value_counts
)
from src.analysis.hypothesis_testing import (
    independent_ttest, paired_ttest, one_sample_ttest,
    chi_square_test, chi_square_independence, one_way_anova
)
from src.modeling.regression import LinearRegression, LogisticRegression
from src.modeling.clustering import KMeans, DBSCAN
from src.preprocessing.cleaner import (
    handle_missing_values, remove_outliers, normalize,
    detect_missing_summary, clip_values
)


# ---------- EDA Tests ----------

class TestDescriptiveStats:
    def test_basic_stats(self):
        data = np.array([1, 2, 3, 4, 5])
        stats = descriptive_stats(data)
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["count"] == 5

    def test_iqr_computation(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stats = descriptive_stats(data)
        assert stats["q1"] == pytest.approx(3.25, abs=0.1)
        assert stats["q3"] == pytest.approx(7.75, abs=0.1)
        assert stats["iqr"] == pytest.approx(4.5, abs=0.2)

    def test_empty_array_raises(self):
        with pytest.raises(ValueError):
            descriptive_stats(np.array([]))

    def test_single_value(self):
        data = np.array([5.0])
        stats = descriptive_stats(data)
        assert stats["mean"] == 5.0
        assert stats["std"] == 0.0


class TestColumnStats:
    def test_two_columns(self):
        data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
        result = column_stats(data, ["A", "B"])
        assert "A" in result
        assert "B" in result
        assert result["A"]["mean"] == 2.5
        assert result["B"]["mean"] == 25.0

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            column_stats(np.array([1, 2, 3]))


class TestFrequencyDistribution:
    def test_histogram_output(self):
        data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
        result = frequency_distribution(data, bins=5)
        assert "counts" in result
        assert "bin_edges" in result
        assert "bin_centers" in result
        assert len(result["counts"]) == 5
        assert len(result["bin_edges"]) == 6


class TestCorrelationMatrix:
    def test_perfect_correlation(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = x * 2
        data = np.column_stack([x, y])
        result = correlation_matrix(data, ["X", "Y"])
        assert result["matrix"][0, 1] == pytest.approx(1.0, abs=1e-10)


class TestOutlierDetection:
    def test_detect_outliers(self):
        data = np.array([1, 2, 3, 4, 5, 100])
        mask, outliers = detect_outliers_iqr(data)
        assert 100 in outliers
        assert len(outliers) >= 1


class TestValueCounts:
    def test_counts(self):
        data = np.array([1, 2, 2, 3, 3, 3])
        counts = value_counts(data)
        assert counts["3"] == 3
        assert counts["2"] == 2
        assert counts["1"] == 1


# ---------- Hypothesis Testing Tests ----------

class TestTTest:
    def test_same_distribution(self):
        rng = np.random.RandomState(42)
        g1 = rng.normal(50, 10, 100)
        g2 = rng.normal(50, 10, 100)
        result = independent_ttest(g1, g2)
        assert result["reject_null"] is False

    def test_different_distributions(self):
        rng = np.random.RandomState(42)
        g1 = rng.normal(50, 5, 200)
        g2 = rng.normal(60, 5, 200)
        result = independent_ttest(g1, g2)
        assert result["reject_null"] is True

    def test_paired_ttest(self):
        before = np.array([5, 6, 7, 8, 9, 10.0])
        after = np.array([6, 7, 8, 9, 10, 11.0])
        result = paired_ttest(before, after)
        assert "p_value" in result

    def test_one_sample_ttest(self):
        data = np.array([5.1, 4.9, 5.0, 5.2, 4.8, 5.1])
        result = one_sample_ttest(data, 5.0)
        assert result["reject_null"] is False


class TestChiSquare:
    def test_uniform_distribution(self):
        observed = np.array([25, 25, 25, 25])
        result = chi_square_test(observed)
        assert result["reject_null"] is False

    def test_non_uniform_distribution(self):
        observed = np.array([90, 5, 3, 2])
        result = chi_square_test(observed)
        assert result["reject_null"] is True

    def test_independence(self):
        table = np.array([[50, 30], [20, 100]])
        result = chi_square_independence(table)
        assert result["reject_null"] is True


class TestANOVA:
    def test_similar_groups(self):
        rng = np.random.RandomState(42)
        g1 = rng.normal(50, 5, 50)
        g2 = rng.normal(50, 5, 50)
        g3 = rng.normal(50, 5, 50)
        result = one_way_anova(g1, g2, g3)
        assert result["reject_null"] is False

    def test_different_groups(self):
        rng = np.random.RandomState(42)
        g1 = rng.normal(30, 3, 50)
        g2 = rng.normal(50, 3, 50)
        g3 = rng.normal(70, 3, 50)
        result = one_way_anova(g1, g2, g3)
        assert result["reject_null"] is True


# ---------- Regression Tests ----------

class TestLinearRegression:
    def test_perfect_fit(self):
        X = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 * X + 1
        model = LinearRegression()
        model.fit(X, y)
        assert model.score(X, y) == pytest.approx(1.0, abs=1e-6)
        params = model.get_params()
        assert params["intercept"] == pytest.approx(1.0, abs=1e-6)
        assert params["coefficients"][0] == pytest.approx(2.0, abs=1e-6)

    def test_prediction(self):
        X = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 3 * X + 2
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(np.array([6.0]))
        assert pred[0] == pytest.approx(20.0, abs=1e-4)

    def test_not_fitted_error(self):
        model = LinearRegression()
        with pytest.raises(RuntimeError):
            model.predict(np.array([1.0]))


class TestLogisticRegression:
    def test_binary_classification(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.normal(-2, 1, (100, 2)), rng.normal(2, 1, (100, 2))])
        y = np.array([0] * 100 + [1] * 100, dtype=float)
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy > 0.85

    def test_predict_proba_range(self):
        X = np.array([[1, 2], [3, 4], [-1, -2]], dtype=float)
        y = np.array([0, 1, 0], dtype=float)
        model = LogisticRegression(n_iterations=100)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)


# ---------- Clustering Tests ----------

class TestKMeans:
    def test_three_clusters(self):
        rng = np.random.RandomState(42)
        c1 = rng.normal([0, 0], 0.5, (50, 2))
        c2 = rng.normal([5, 5], 0.5, (50, 2))
        c3 = rng.normal([10, 0], 0.5, (50, 2))
        X = np.vstack([c1, c2, c3])
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(X)
        assert len(set(km.labels)) == 3
        assert km.inertia > 0

    def test_predict(self):
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]], dtype=float)
        km = KMeans(n_clusters=2, random_state=42)
        km.fit(X)
        labels = km.predict(np.array([[0.5, 0.5], [10.5, 10.5]]))
        assert labels[0] != labels[1]


class TestDBSCAN:
    def test_two_clusters(self):
        rng = np.random.RandomState(42)
        c1 = rng.normal([0, 0], 0.3, (50, 2))
        c2 = rng.normal([5, 5], 0.3, (50, 2))
        X = np.vstack([c1, c2])
        db = DBSCAN(eps=1.0, min_samples=5)
        db.fit(X)
        params = db.get_params()
        assert params["n_clusters"] == 2

    def test_noise_detection(self):
        X = np.array([[0, 0], [0.1, 0], [0, 0.1], [10, 10]], dtype=float)
        db = DBSCAN(eps=0.5, min_samples=2)
        db.fit(X)
        assert db.labels[-1] == -1


# ---------- Preprocessing Tests ----------

class TestMissingValues:
    def test_mean_imputation(self):
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = handle_missing_values(data, strategy="mean")
        assert not np.any(np.isnan(result))
        assert result[2] == pytest.approx(3.0, abs=0.01)

    def test_drop_strategy(self):
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = handle_missing_values(data, strategy="drop")
        assert len(result) == 3

    def test_constant_imputation(self):
        data = np.array([1.0, np.nan, 3.0])
        result = handle_missing_values(data, strategy="constant", fill_value=0.0)
        assert result[1] == 0.0


class TestNormalization:
    def test_minmax_range(self):
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        result, params = normalize(data, method="minmax")
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_standard_mean_zero(self):
        data = np.array([10, 20, 30, 40, 50], dtype=float)
        result, params = normalize(data, method="standard")
        assert np.mean(result) == pytest.approx(0.0, abs=1e-10)


class TestRemoveOutliers:
    def test_iqr_removes_extreme(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float)
        result = remove_outliers(data, method="iqr")
        assert 100 not in result


class TestMissingSummary:
    def test_summary(self):
        data = np.array([[1, np.nan], [2, 3], [np.nan, 4], [5, 6]], dtype=float)
        summary = detect_missing_summary(data, ["A", "B"])
        assert summary["A"]["n_missing"] == 1
        assert summary["B"]["n_missing"] == 1


class TestClipValues:
    def test_clip(self):
        data = np.array([-5, 0, 5, 10, 15], dtype=float)
        result = clip_values(data, lower=0, upper=10)
        assert result.min() == 0
        assert result.max() == 10
