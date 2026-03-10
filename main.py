"""
Data Science Portfolio - Main Demo

Demonstrates the full data science pipeline using a synthetic dataset:
data generation, cleaning, EDA, modeling, and visualization.
"""

import numpy as np
from src.analysis.eda import descriptive_stats, correlation_matrix, frequency_distribution, detect_outliers_iqr
from src.analysis.hypothesis_testing import independent_ttest, one_way_anova, chi_square_test
from src.modeling.regression import LinearRegression, LogisticRegression
from src.modeling.clustering import KMeans, DBSCAN
from src.preprocessing.cleaner import handle_missing_values, normalize, remove_outliers
from src.visualization.plots import (
    plot_histogram, plot_scatter, plot_correlation_heatmap,
    plot_boxplot, plot_bar
)


def generate_synthetic_dataset(n_samples: int = 500, random_state: int = 42) -> dict:
    """Generate a synthetic dataset for demonstration."""
    rng = np.random.RandomState(random_state)

    age = rng.normal(35, 10, n_samples).clip(18, 70)
    income = age * 1500 + rng.normal(0, 10000, n_samples)
    income = income.clip(15000, None)
    experience = (age - 18) * 0.7 + rng.normal(0, 3, n_samples)
    experience = experience.clip(0, None)
    satisfaction = 3 + 0.02 * income / 1000 + rng.normal(0, 1, n_samples)
    satisfaction = satisfaction.clip(1, 10)

    nan_indices = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    income_with_nan = income.copy()
    income_with_nan[nan_indices] = np.nan

    purchased = (0.5 * (income > 50000).astype(float) +
                 0.3 * (age > 30).astype(float) +
                 0.2 * rng.random(n_samples))
    purchased = (purchased > 0.5).astype(int)

    return {
        "age": age,
        "income": income,
        "income_with_nan": income_with_nan,
        "experience": experience,
        "satisfaction": satisfaction,
        "purchased": purchased,
        "n_samples": n_samples,
    }


def demo_preprocessing(dataset: dict) -> dict:
    """Demonstrate data preprocessing capabilities."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PREPROCESSING")
    print("=" * 60)

    income_clean = handle_missing_values(dataset["income_with_nan"], strategy="median")
    n_original_nan = int(np.sum(np.isnan(dataset["income_with_nan"])))
    n_after_nan = int(np.sum(np.isnan(income_clean)))
    print(f"\nMissing value imputation (median):")
    print(f"  Before: {n_original_nan} NaN values")
    print(f"  After:  {n_after_nan} NaN values")

    income_no_outliers = remove_outliers(income_clean, method="iqr")
    print(f"\nOutlier removal (IQR method):")
    print(f"  Before: {len(income_clean)} observations")
    print(f"  After:  {len(income_no_outliers)} observations")
    print(f"  Removed: {len(income_clean) - len(income_no_outliers)} outliers")

    features = np.column_stack([dataset["age"], income_clean, dataset["experience"]])
    normalized, params = normalize(features, method="standard")
    print(f"\nStandardization applied to {features.shape[1]} features")
    print(f"  Feature means after: {normalized.mean(axis=0).round(6)}")
    print(f"  Feature stds after:  {normalized.std(axis=0, ddof=1).round(4)}")

    dataset["income_clean"] = income_clean
    dataset["features_normalized"] = normalized
    return dataset


def demo_eda(dataset: dict):
    """Demonstrate exploratory data analysis."""
    print("\n" + "=" * 60)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    for name, data in [("Age", dataset["age"]), ("Income", dataset["income_clean"])]:
        stats = descriptive_stats(data)
        print(f"\n--- {name} ---")
        print(f"  Mean: {stats['mean']:.2f}  |  Median: {stats['median']:.2f}")
        print(f"  Std:  {stats['std']:.2f}  |  IQR: {stats['iqr']:.2f}")
        print(f"  Min:  {stats['min']:.2f}  |  Max: {stats['max']:.2f}")
        print(f"  Skewness: {stats['skewness']:.4f}  |  Kurtosis: {stats['kurtosis']:.4f}")

    features = np.column_stack([dataset["age"], dataset["income_clean"],
                                dataset["experience"], dataset["satisfaction"]])
    col_names = ["Age", "Income", "Experience", "Satisfaction"]
    corr = correlation_matrix(features, col_names)
    print("\n--- Correlation Matrix ---")
    for i, row_name in enumerate(col_names):
        vals = " | ".join(f"{corr['matrix'][i, j]:+.3f}" for j in range(len(col_names)))
        print(f"  {row_name:>12}: [{vals}]")

    outlier_mask, outlier_vals = detect_outliers_iqr(dataset["income_clean"])
    print(f"\nOutlier detection (IQR) on Income: {len(outlier_vals)} outliers found")


def demo_hypothesis_testing(dataset: dict):
    """Demonstrate hypothesis testing."""
    print("\n" + "=" * 60)
    print("PHASE 3: HYPOTHESIS TESTING")
    print("=" * 60)

    young = dataset["income_clean"][dataset["age"] < 35]
    old = dataset["income_clean"][dataset["age"] >= 35]
    result = independent_ttest(young, old)
    print(f"\nIndependent t-test: Young vs Old income")
    print(f"  t-statistic: {result['t_statistic']:.4f}")
    print(f"  p-value:     {result['p_value']:.6f}")
    print(f"  Reject H0:   {result['reject_null']}")

    low = dataset["satisfaction"][dataset["income_clean"] < 40000]
    mid = dataset["satisfaction"][(dataset["income_clean"] >= 40000) & (dataset["income_clean"] < 60000)]
    high = dataset["satisfaction"][dataset["income_clean"] >= 60000]
    if len(low) >= 2 and len(mid) >= 2 and len(high) >= 2:
        anova_result = one_way_anova(low, mid, high)
        print(f"\nOne-way ANOVA: Satisfaction by Income group")
        print(f"  F-statistic: {anova_result['f_statistic']:.4f}")
        print(f"  p-value:     {anova_result['p_value']:.6f}")
        print(f"  Reject H0:   {anova_result['reject_null']}")
        print(f"  Group means:  {[f'{m:.2f}' for m in anova_result['group_means']]}")

    observed = np.array([45, 55, 30, 70])
    chi_result = chi_square_test(observed)
    print(f"\nChi-square goodness-of-fit:")
    print(f"  Chi2:     {chi_result['chi2_statistic']:.4f}")
    print(f"  p-value:  {chi_result['p_value']:.6f}")
    print(f"  Reject H0: {chi_result['reject_null']}")


def demo_modeling(dataset: dict):
    """Demonstrate regression and clustering models."""
    print("\n" + "=" * 60)
    print("PHASE 4: MODELING")
    print("=" * 60)

    X = dataset["age"].reshape(-1, 1)
    y = dataset["income_clean"]
    lr = LinearRegression()
    lr.fit(X, y)
    params = lr.get_params()
    r2 = lr.score(X, y)
    print(f"\n--- Linear Regression: Income ~ Age ---")
    print(f"  Intercept:    {params['intercept']:.2f}")
    print(f"  Coefficient:  {params['coefficients'][0]:.2f}")
    print(f"  R-squared:    {r2:.4f}")

    X_log = np.column_stack([dataset["age"], dataset["income_clean"]])
    X_norm, _ = normalize(X_log, method="standard")
    y_log = dataset["purchased"]
    log_reg = LogisticRegression(learning_rate=0.1, n_iterations=500)
    log_reg.fit(X_norm, y_log)
    accuracy = log_reg.score(X_norm, y_log)
    print(f"\n--- Logistic Regression: Purchased ~ Age + Income ---")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  Iterations:   {len(log_reg.loss_history)}")

    features_2d = np.column_stack([dataset["age"], dataset["income_clean"]])
    features_norm, _ = normalize(features_2d, method="standard")
    km = KMeans(n_clusters=3, random_state=42)
    km.fit(features_norm)
    km_params = km.get_params()
    print(f"\n--- K-Means Clustering (k=3) ---")
    print(f"  Inertia:    {km_params['inertia']:.4f}")
    for i, c in enumerate(km_params["centroids"]):
        print(f"  Centroid {i}: [{c[0]:.3f}, {c[1]:.3f}]")

    db = DBSCAN(eps=0.5, min_samples=5)
    db.fit(features_norm)
    db_params = db.get_params()
    print(f"\n--- DBSCAN Clustering ---")
    print(f"  Clusters found: {db_params['n_clusters']}")
    print(f"  Noise points:   {db_params['n_noise_points']}")


def demo_visualization(dataset: dict):
    """Demonstrate visualization capabilities."""
    print("\n" + "=" * 60)
    print("PHASE 5: VISUALIZATION")
    print("=" * 60)

    fig1 = plot_histogram(dataset["age"], title="Age Distribution", xlabel="Age", ylabel="Count")
    print("\n  [Generated] Age distribution histogram")

    fig2 = plot_scatter(dataset["age"], dataset["income_clean"],
                        title="Age vs Income", xlabel="Age", ylabel="Income",
                        add_trendline=True)
    print("  [Generated] Age vs Income scatter plot with trendline")

    features = np.column_stack([dataset["age"], dataset["income_clean"],
                                dataset["experience"], dataset["satisfaction"]])
    corr = np.corrcoef(features, rowvar=False)
    fig3 = plot_correlation_heatmap(corr, labels=["Age", "Income", "Experience", "Satisfaction"])
    print("  [Generated] Correlation heatmap")

    young = dataset["income_clean"][dataset["age"] < 30]
    mid = dataset["income_clean"][(dataset["age"] >= 30) & (dataset["age"] < 45)]
    senior = dataset["income_clean"][dataset["age"] >= 45]
    fig4 = plot_boxplot([young, mid, senior], labels=["Young (<30)", "Mid (30-45)", "Senior (45+)"],
                        title="Income by Age Group")
    print("  [Generated] Income by age group box plot")

    categories = ["Low", "Medium", "High", "Premium"]
    values = np.array([120, 250, 180, 80])
    fig5 = plot_bar(categories, values, title="Customer Segments",
                    xlabel="Segment", ylabel="Count")
    print("  [Generated] Customer segments bar chart")

    import matplotlib.pyplot as plt
    plt.close("all")
    print("\n  All figures generated successfully (closed to free memory)")


def main():
    """Run the full data science demo pipeline."""
    print("=" * 60)
    print("  DATA SCIENCE PORTFOLIO - PIPELINE DEMO")
    print("=" * 60)

    dataset = generate_synthetic_dataset(n_samples=500)
    print(f"\nSynthetic dataset generated: {dataset['n_samples']} samples")
    print(f"  Features: age, income, experience, satisfaction")
    print(f"  Target: purchased (binary)")

    dataset = demo_preprocessing(dataset)
    demo_eda(dataset)
    demo_hypothesis_testing(dataset)
    demo_modeling(dataset)
    demo_visualization(dataset)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
