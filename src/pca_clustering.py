import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    AffinityPropagation,
    Birch,
    BisectingKMeans,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.gridspec import GridSpec
from sklearn.utils.estimator_checks import check_estimator

from kneed import KneeLocator


def kmeans_clustering_analysis(
    data,
    max_clusters=15,
    n_splits=3,
    random_state=42,
    num_clusters=None,
    normalize=False,
):
    """
    Perform K-Means clustering analysis with cross-validation and automatic cluster selection.

    Parameters:
    data (DataFrame/array): Input data
    max_clusters (int): Maximum number of clusters to evaluate
    n_splits (int): Number of cross-validation splits
    random_state (int): Random state for reproducibility

    Returns:
    dict: Dictionary containing results, metrics, and optimal clusters
    """
    # Standardize the data
    if normalize:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.array(data))
    else:
        scaled_data = np.array(data)

    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize results dictionary
    results = {
        "n_clusters": [],
        "silhouette_score": [],
        "silhouette_std": [],
        "calinski_score": [],
        "calinski_std": [],
        "inertia": [],
    }

    # Evaluate different numbers of clusters with cross-validation
    for n in range(2, max_clusters + 1):
        silhouette_scores = []
        calinski_scores = []
        inertia_scores = []

        for train_idx, val_idx in kf.split(scaled_data):
            # Split data
            train_data = scaled_data[train_idx]
            val_data = scaled_data[val_idx]

            # Fit K-Means clustering
            kmeans = KMeans(n_clusters=n, random_state=random_state, algorithm="elkan")
            kmeans.fit(train_data)
            train_labels = kmeans.predict(train_data)
            val_labels = kmeans.predict(val_data)

            # Calculate metrics
            if len(np.unique(val_labels)) > 1:  # Check if more than one cluster
                silhouette_scores.append(silhouette_score(val_data, val_labels))
                calinski_scores.append(calinski_harabasz_score(val_data, val_labels))

            # Calculate inertia
            inertia_scores.append(kmeans.inertia_)

        # Store results
        results["n_clusters"].append(n)
        results["silhouette_score"].append(np.mean(silhouette_scores))
        results["silhouette_std"].append(np.std(silhouette_scores))
        results["calinski_score"].append(np.mean(calinski_scores))
        results["calinski_std"].append(np.std(calinski_scores))
        results["inertia"].append(np.mean(inertia_scores))

    # Find optimal number of clusters
    if num_clusters:
        optimal_clusters = num_clusters
    else:
        optimal_clusters = find_optimal_clusters(results)
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=random_state)
    final_labels = final_kmeans.fit_predict(scaled_data)

    # Get centroids
    centroids = final_kmeans.cluster_centers_

    if normalize:
        centroids = scaler.inverse_transform(centroids)

    if isinstance(data, pd.DataFrame):
        centroids = pd.DataFrame(centroids, columns=data.columns)

    # Create visualizations
    plot_kmeans_clustering_results(results, optimal_clusters)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return {
        "results_df": results_df,
        "optimal_clusters": optimal_clusters,
        "labels": final_labels,
        "final_model": final_kmeans,
        "centroids": centroids,
    }


def plot_kmeans_clustering_results(results, optimal_clusters):
    """
    Create visualization of clustering results.
    """
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Plot evaluation metrics with error bars
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])

    # Silhouette and Calinski-Harabasz scores
    ax1.errorbar(
        results["n_clusters"],
        results["silhouette_score"],
        yerr=results["silhouette_std"],
        fmt="bo-",
        label="Silhouette",
    )
    ax1_twin = ax1.twinx()
    ax1_twin.errorbar(
        results["n_clusters"],
        results["calinski_score"],
        yerr=results["calinski_std"],
        fmt="ro-",
        label="Calinski-Harabasz",
    )

    # Highlight optimal clusters
    ax1.axvline(
        x=optimal_clusters,
        color="g",
        linestyle="--",
        label=f"Optimal (n={optimal_clusters})",
    )

    ax1.set_title("Clustering Evaluation Metrics")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Silhouette Score", color="b")
    ax1_twin.set_ylabel("Calinski-Harabasz Score", color="r")

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Elbow plot (inertia)
    ax2.plot(results["n_clusters"], results["inertia"], "go-")
    ax2.axvline(x=optimal_clusters, color="r", linestyle="--")
    ax2.set_title("Elbow Plot (Inertia)")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Inertia (Sum of Squared Distances)")
    ax2.set_ylim(0, None)

    # Cross-validation stability plot
    ax3.fill_between(
        results["n_clusters"],
        np.array(results["silhouette_score"]) - np.array(results["silhouette_std"]),
        np.array(results["silhouette_score"]) + np.array(results["silhouette_std"]),
        alpha=0.2,
    )
    ax3.plot(results["n_clusters"], results["silhouette_score"], "b-")
    ax3.set_title("Cross-validation Stability")
    ax3.set_xlabel("Number of Clusters")
    ax3.set_ylabel("Silhouette Score with Std Dev")

    plt.tight_layout()
    plt.show()


def hierarchical_clustering_analysis(
    data,
    max_clusters=15,
    n_splits=3,
    random_state=42,
    num_clusters=None,
    normalize=False,
):
    """
    Perform hierarchical clustering analysis with cross-validation and automatic cluster selection.

    Parameters:
    data (DataFrame/array): Input data
    max_clusters (int): Maximum number of clusters to evaluate
    n_splits (int): Number of cross-validation splits
    random_state (int): Random state for reproducibility

    Returns:
    dict: Dictionary containing results, metrics, and optimal clusters
    """
    # Standardize the data
    if normalize:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.array(data))
    else:
        scaled_data = np.array(data)

    # Calculate linkage matrix for dendrogram
    linkage_matrix = linkage(scaled_data, method="ward")

    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize results dictionary
    results = {
        "n_clusters": [],
        "silhouette_score": [],
        "silhouette_std": [],
        "calinski_score": [],
        "calinski_std": [],
        "inertia": [],
    }

    # Evaluate different numbers of clusters with cross-validation
    for n in range(2, max_clusters + 1):
        silhouette_scores = []
        calinski_scores = []
        inertia_scores = []

        for train_idx, val_idx in kf.split(scaled_data):
            # Split data
            train_data = scaled_data[train_idx]
            val_data = scaled_data[val_idx]

            # Fit clustering
            clustering = AgglomerativeClustering(n_clusters=n, linkage="ward")
            train_labels = clustering.fit_predict(train_data)
            val_labels = clustering.fit_predict(val_data)

            # Calculate metrics
            if len(np.unique(val_labels)) > 1:  # Check if more than one cluster
                silhouette_scores.append(silhouette_score(val_data, val_labels))
                calinski_scores.append(calinski_harabasz_score(val_data, val_labels))

            # Calculate inertia
            inertia = 0
            for i in range(n):
                cluster_points = val_data[val_labels == i]
                if len(cluster_points) > 0:
                    centroid = cluster_points.mean(axis=0)
                    inertia += np.sum((cluster_points - centroid) ** 2)
            inertia_scores.append(inertia)

        # Store results
        results["n_clusters"].append(n)
        results["silhouette_score"].append(np.mean(silhouette_scores))
        results["silhouette_std"].append(np.std(silhouette_scores))
        results["calinski_score"].append(np.mean(calinski_scores))
        results["calinski_std"].append(np.std(calinski_scores))
        results["inertia"].append(np.mean(inertia_scores))

    # Find optimal number of clusters
    if num_clusters:
        optimal_clusters = num_clusters
    else:
        optimal_clusters = find_optimal_clusters(results)
    final_clustering = AgglomerativeClustering(
        n_clusters=optimal_clusters, linkage="ward"
    )
    final_labels = final_clustering.fit_predict(scaled_data)

    cluster_sizes = np.bincount(final_labels)
    if np.min(cluster_sizes) < 1:
        # Reduce number of clusters until minimum size is met
        while optimal_clusters > 2:
            optimal_clusters -= 1
            final_clustering = AgglomerativeClustering(
                n_clusters=optimal_clusters, linkage="ward"
            )
            final_labels = final_clustering.fit_predict(scaled_data)
            cluster_sizes = np.bincount(final_labels)
            if np.min(cluster_sizes) >= 1:
                break

    centroids = np.array(
        [scaled_data[final_labels == i].mean(axis=0) for i in range(optimal_clusters)]
    )
    if normalize:
        centroids = scaler.inverse_transform(centroids)

    if isinstance(data, pd.DataFrame):
        centroids = pd.DataFrame(centroids, columns=data.columns)

    # Create visualizations
    plot_hirachical_clustering_results(results, linkage_matrix, optimal_clusters)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return {
        "results_df": results_df,
        "linkage_matrix": linkage_matrix,
        "optimal_clusters": optimal_clusters,
        "labels": final_labels,
        "final_model": final_clustering,
        "centroids": centroids,
    }


def plot_hirachical_clustering_results(results, linkage_matrix, optimal_clusters):
    """
    Create visualization of clustering results.
    """
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Plot dendrogram
    ax_dendrogram = fig.add_subplot(gs[0, :])
    dendrogram(linkage_matrix)
    ax_dendrogram.set_title("Hierarchical Clustering Dendrogram")
    ax_dendrogram.set_xlabel("Sample Index")
    ax_dendrogram.set_ylabel("Distance")

    # Plot evaluation metrics with error bars
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])

    # Silhouette and Calinski-Harabasz scores
    ax1.errorbar(
        results["n_clusters"],
        results["silhouette_score"],
        yerr=results["silhouette_std"],
        fmt="bo-",
        label="Silhouette",
    )
    ax1_twin = ax1.twinx()
    ax1_twin.errorbar(
        results["n_clusters"],
        results["calinski_score"],
        yerr=results["calinski_std"],
        fmt="ro-",
        label="Calinski-Harabasz",
    )

    # Highlight optimal clusters
    ax1.axvline(
        x=optimal_clusters,
        color="g",
        linestyle="--",
        label=f"Optimal (n={optimal_clusters})",
    )

    ax1.set_title("Clustering Evaluation Metrics")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Silhouette Score", color="b")
    ax1_twin.set_ylabel("Calinski-Harabasz Score", color="r")

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Elbow plot
    ax2.plot(results["n_clusters"], results["inertia"], "go-")
    ax2.axvline(x=optimal_clusters, color="r", linestyle="--")
    ax2.set_title("Elbow Plot")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Within-cluster Sum of Squares")
    ax2.set_ylim(0, None)

    # Cross-validation stability plot
    ax3.fill_between(
        results["n_clusters"],
        np.array(results["silhouette_score"]) - np.array(results["silhouette_std"]),
        np.array(results["silhouette_score"]) + np.array(results["silhouette_std"]),
        alpha=0.2,
    )
    ax3.plot(results["n_clusters"], results["silhouette_score"], "b-")
    ax3.set_title("Cross-validation Stability")
    ax3.set_xlabel("Number of Clusters")
    ax3.set_ylabel("Silhouette Score with Std Dev")

    plt.tight_layout()
    plt.show()


def get_cluster_labels(data, n_clusters):
    """
    Get cluster labels for a specific number of clusters.

    Parameters:
    data (DataFrame/array): Input data
    n_clusters (int): Number of clusters

    Returns:
    array: Cluster labels
    DataFrame: Cluster centroids
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = data  # scaler.fit_transform(data)

    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(scaled_data)

    # Calculate centroids
    centroids = np.zeros((n_clusters, scaled_data.shape[1]))
    for i in range(n_clusters):
        cluster_points = scaled_data[labels == i]
        centroids[i] = cluster_points.mean(axis=0)

    # Transform centroids back to original scale
    # centroids = scaler.inverse_transform(centroids)

    # Create centroids DataFrame
    if isinstance(data, pd.DataFrame):
        centroids = pd.DataFrame(centroids, columns=data.columns)

    return labels, centroids


def affinity_propagation_analysis(
    data,
    preference_range=None,
    damping=0.5,
    n_splits=5,
    random_state=42,
    num_clusters=None,
    normalize=False,
):
    """
    Perform Affinity Propagation clustering analysis with cross-validation.

    Parameters:
    data: DataFrame/array of features
    preference_range: tuple of (min, max) preferences to try
    damping: damping factor (default 0.5)
    n_splits: number of cross-validation splits (default 5)

    Returns:
    dict: Results, metrics, and optimal clusters
    """
    if normalize:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.array(data))
    else:
        scaled_data = np.array(data)

    if preference_range is None:
        # Set default preference range based on data similarities
        S = -np.sum(np.square(scaled_data[:, None] - scaled_data), axis=2)
        min_pref = np.percentile(S, 1)
        max_pref = np.percentile(S, 99)
        preferences = np.linspace(min_pref, max_pref, 10)
    else:
        preferences = np.linspace(preference_range[0], preference_range[1], 10)

    results = {
        "preference": [],
        "silhouette_score": [],
        "silhouette_std": [],
        "calinski_score": [],
        "calinski_std": [],
        "n_clusters": [],
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for pref in preferences:
        n_clusters_list = []
        silhouette_scores = []
        calinski_scores = []

        for train_idx, val_idx in kf.split(scaled_data):
            train_data = scaled_data[train_idx]
            val_data = scaled_data[val_idx]

            clustering = AffinityPropagation(
                preference=pref, damping=damping, random_state=random_state
            )
            try:
                labels = clustering.fit_predict(train_data)
                n_clusters = len(np.unique(labels))

                if n_clusters > 1:  # Metrics need at least 2 clusters
                    silhouette = silhouette_score(
                        val_data, clustering.predict(val_data)
                    )
                    calinski = calinski_harabasz_score(
                        val_data, clustering.predict(val_data)
                    )
                else:
                    silhouette = 0
                    calinski = 0

                # Collect results for each split
                n_clusters_list.append(n_clusters)
                silhouette_scores.append(silhouette)
                calinski_scores.append(calinski)

            except Exception as e:
                print(f"Clustering failed for preference {pref}: {str(e)}")
                continue

        # Store mean and std for the preference
        results["preference"].append(pref)
        results["silhouette_score"].append(np.mean(silhouette_scores))
        results["silhouette_std"].append(np.std(silhouette_scores))
        results["calinski_score"].append(np.mean(calinski_scores))
        results["calinski_std"].append(np.std(calinski_scores))
        results["n_clusters"].append(np.mean(n_clusters_list))

    # Find optimal number of clusters
    if num_clusters:
        optimal_clusters = num_clusters
    else:
        optimal_clusters = find_ap_optimal_clusters(results)
    optimal_pref = preferences[
        np.argmin(np.abs(np.array(results["n_clusters"]) - optimal_clusters))
    ]

    # Final clustering with optimal preference
    final_clustering = AffinityPropagation(
        preference=optimal_pref, damping=damping, random_state=1
    )
    final_labels = final_clustering.fit_predict(scaled_data)

    # Plotting results
    plot_affinity_propagation_results(results, optimal_clusters)

    return {
        "results_df": pd.DataFrame(results),
        "optimal_clusters": optimal_clusters,
        "optimal_preference": optimal_pref,
        "labels": final_labels,
        "final_model": final_clustering,
    }


def find_ap_optimal_clusters(results):
    """
    Find optimal number of clusters using multiple metrics and proper normalization.
    """
    n_clusters = np.array(results["n_clusters"])

    # Normalize metrics between 0 and 1
    silhouette = np.array(results["silhouette_score"])
    silhouette_norm = (silhouette - silhouette.min()) / (
        silhouette.max() - silhouette.min()
    )

    calinski = np.array(results["calinski_score"])
    calinski_norm = (calinski - calinski.min()) / (calinski.max() - calinski.min())

    # Weighted combination of metrics
    weights = {"silhouette": 0.50, "calinski": 0.50}

    combined_score = (
        weights["silhouette"] * silhouette_norm + weights["calinski"] * calinski_norm
    )

    optimal_idx = np.argmax(combined_score)
    return n_clusters[optimal_idx]


def plot_affinity_propagation_results(results, optimal_clusters):
    """
    Plot Affinity Propagation results including metrics and optimal clusters.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Number of clusters vs preference
    ax1 = axes[0, 0]
    ax1.errorbar(
        results["preference"],
        results["n_clusters"],
        # yerr=results["n_clusters_std"],
        fmt="bo-",
    )
    ax1.set_title("Number of Clusters vs Preference")
    ax1.set_xlabel("Preference")
    ax1.set_ylabel("Number of Clusters")
    ax1.axhline(y=optimal_clusters, color="g", linestyle="--", label="Optimal Clusters")
    ax1.legend()

    # Silhouette Score vs Number of Clusters
    ax2 = axes[0, 1]
    ax2.errorbar(
        results["n_clusters"],
        results["silhouette_score"],
        yerr=results["silhouette_std"],
        fmt="ro-",
    )
    ax2.set_title("Silhouette Score vs Number of Clusters")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Silhouette Score")

    # Calinski-Harabasz Score vs Number of Clusters
    ax3 = axes[1, 0]
    ax3.errorbar(
        results["n_clusters"],
        results["calinski_score"],
        yerr=results["calinski_std"],
        fmt="go-",
    )
    ax3.set_title("Calinski-Harabasz Score vs Number of Clusters")
    ax3.set_xlabel("Number of Clusters")
    ax3.set_ylabel("Calinski-Harabasz Score")

    # Elbow plot: Preference vs Silhouette Score
    ax4 = axes[1, 1]
    ax4.plot(results["preference"], results["silhouette_score"], "mo-")
    ax4.set_title("Elbow Plot: Preference vs Silhouette Score")
    ax4.set_xlabel("Preference")
    ax4.set_ylabel("Silhouette Score")
    ax4.set_ylim(0, None)

    plt.tight_layout()
    plt.show()


def get_ap_cluster_labels(data, preference=None, damping=0.5, random_state=42):
    """
    Get cluster labels using Affinity Propagation.

    Parameters:
    data: DataFrame/array of features
    preference: preference parameter for AP
    damping: damping factor

    Returns:
    array: Cluster labels
    array: Cluster centers indices
    """
    clustering = AffinityPropagation(
        preference=preference, damping=damping, random_state=random_state
    )
    labels = clustering.fit_predict(data)

    # Get cluster centers in original scale
    centers = clustering.cluster_centers_

    if isinstance(data, pd.DataFrame):
        centers = pd.DataFrame(centers, columns=data.columns)

    return labels, centers, clustering.cluster_centers_indices_


def gaussian_mixture_clustering_analysis(
    data,
    max_clusters=15,
    n_splits=5,
    random_state=42,
    num_clusters=None,
    normalize=False,
):
    """
    Perform Gaussian Mixture Model clustering analysis with cross-validation and automatic cluster selection.

    Parameters:
    data (DataFrame/array): Input data
    max_clusters (int): Maximum number of clusters to evaluate
    n_splits (int): Number of cross-validation splits
    random_state (int): Random state for reproducibility

    Returns:
    dict: Dictionary containing results, metrics, and optimal clusters
    """
    if normalize:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.array(data))
    else:
        scaled_data = np.array(data)

    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize results dictionary
    results = {
        "n_clusters": [],
        "silhouette_score": [],
        "silhouette_std": [],
        "calinski_score": [],
        "calinski_std": [],
        "inertia": [],
    }

    # Evaluate different numbers of clusters with cross-validation
    for n in range(2, max_clusters + 1):
        silhouette_scores = []
        calinski_scores = []
        inertia_scores = []

        for train_idx, val_idx in kf.split(scaled_data):
            # Split data
            train_data = scaled_data[train_idx]
            val_data = scaled_data[val_idx]

            # Fit GMM clustering
            gmm = GaussianMixture(n_components=n, random_state=random_state)
            gmm.fit(train_data)
            train_labels = gmm.predict(train_data)
            val_labels = gmm.predict(val_data)

            # Calculate metrics
            if len(np.unique(val_labels)) > 1:  # Check if more than one cluster
                silhouette_scores.append(silhouette_score(val_data, val_labels))
                calinski_scores.append(calinski_harabasz_score(val_data, val_labels))

            # Calculate inertia (using log-likelihood from GMM)
            inertia_scores.append(
                -gmm.score(val_data)
            )  # Inertia is the negative log-likelihood

        # Store results
        results["n_clusters"].append(n)
        results["silhouette_score"].append(np.mean(silhouette_scores))
        results["silhouette_std"].append(np.std(silhouette_scores))
        results["calinski_score"].append(np.mean(calinski_scores))
        results["calinski_std"].append(np.std(calinski_scores))
        results["inertia"].append(np.mean(inertia_scores))

    # Find optimal number of clusters
    if num_clusters:
        optimal_clusters = num_clusters
    else:
        optimal_clusters = find_optimal_clusters(results)
    final_gmm = GaussianMixture(
        n_components=optimal_clusters, random_state=random_state
    )
    final_labels = final_gmm.fit_predict(scaled_data)

    # Calculate centroids (for GMM, we use the means of the components as the centroids)
    centroids = final_gmm.means_
    if normalize:
        centroids = scaler.inverse_transform(centroids)

    if isinstance(data, pd.DataFrame):
        centroids = pd.DataFrame(centroids, columns=data.columns)

    # Create visualizations
    plot_GMM_clustering_results(results, optimal_clusters)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return {
        "results_df": results_df,
        "optimal_clusters": optimal_clusters,
        "labels": final_labels,
        "final_model": final_gmm,
        "centroids": centroids,
    }


def plot_GMM_clustering_results(
    results,
    optimal_clusters,
):
    """
    Create visualization of clustering results.
    """
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Plot evaluation metrics with error bars
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])

    # Silhouette and Calinski-Harabasz scores
    ax1.errorbar(
        results["n_clusters"],
        results["silhouette_score"],
        yerr=results["silhouette_std"],
        fmt="bo-",
        label="Silhouette",
    )
    ax1_twin = ax1.twinx()
    ax1_twin.errorbar(
        results["n_clusters"],
        results["calinski_score"],
        yerr=results["calinski_std"],
        fmt="ro-",
        label="Calinski-Harabasz",
    )

    # Highlight optimal clusters
    ax1.axvline(
        x=optimal_clusters,
        color="g",
        linestyle="--",
        label=f"Optimal (n={optimal_clusters})",
    )

    ax1.set_title("Clustering Evaluation Metrics")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Silhouette Score", color="b")
    ax1_twin.set_ylabel("Calinski-Harabasz Score", color="r")

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Elbow plot (inertia)
    ax2.plot(results["n_clusters"], results["inertia"], "go-")
    ax2.axvline(x=optimal_clusters, color="r", linestyle="--")
    ax2.set_title("Elbow Plot (Inertia)")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Inertia (Negative Log-Likelihood)")
    ax2.set_ylim(0, None)

    # Cross-validation stability plot
    ax3.fill_between(
        results["n_clusters"],
        np.array(results["silhouette_score"]) - np.array(results["silhouette_std"]),
        np.array(results["silhouette_score"]) + np.array(results["silhouette_std"]),
        alpha=0.2,
    )
    ax3.plot(results["n_clusters"], results["silhouette_score"], "b-")
    ax3.set_title("Cross-validation Stability")
    ax3.set_xlabel("Number of Clusters")
    ax3.set_ylabel("Silhouette Score with Std Dev")

    plt.tight_layout()
    plt.show()


def birch_clustering_analysis(
    data,
    max_clusters=15,
    n_splits=5,
    random_state=42,
    num_clusters=None,
    normalize=False,
):
    """
    Perform BIRCH clustering analysis with cross-validation and automatic cluster selection.

    Parameters:
    data (DataFrame/array): Input data
    max_clusters (int): Maximum number of clusters to evaluate
    n_splits (int): Number of cross-validation splits
    random_state (int): Random state for reproducibility

    Returns:
    dict: Dictionary containing results, metrics, and optimal clusters
    """
    # Standardize the data
    if normalize:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.array(data))
    else:
        scaled_data = np.array(data)

    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize results dictionary
    results = {
        "n_clusters": [],
        "silhouette_score": [],
        "silhouette_std": [],
        "calinski_score": [],
        "calinski_std": [],
        "inertia": [],
    }

    # Evaluate different numbers of clusters with cross-validation
    for n in range(2, max_clusters + 1):
        silhouette_scores = []
        calinski_scores = []
        inertia_scores = []

        for train_idx, val_idx in kf.split(scaled_data):
            # Split data
            train_data = scaled_data[train_idx]
            val_data = scaled_data[val_idx]

            # Fit BIRCH clustering
            birch = Birch(n_clusters=n)
            birch.fit(train_data)
            train_labels = birch.predict(train_data)
            val_labels = birch.predict(val_data)

            # Calculate metrics
            if len(np.unique(val_labels)) > 1:  # Check if more than one cluster
                silhouette_scores.append(silhouette_score(val_data, val_labels))
                calinski_scores.append(calinski_harabasz_score(val_data, val_labels))

            # Calculate inertia (using sum of squared distances to centroids)
            inertia = 0
            centroids = birch.subcluster_centers_
            for i in range(n):
                cluster_points = val_data[val_labels == i]
                if len(cluster_points) > 0:
                    centroid = centroids[i]
                    inertia += np.sum((cluster_points - centroid) ** 2)
            inertia_scores.append(inertia)

        # Store results
        results["n_clusters"].append(n)
        results["silhouette_score"].append(np.mean(silhouette_scores))
        results["silhouette_std"].append(np.std(silhouette_scores))
        results["calinski_score"].append(np.mean(calinski_scores))
        results["calinski_std"].append(np.std(calinski_scores))
        results["inertia"].append(np.mean(inertia_scores))

    # Find optimal number of clusters
    if num_clusters:
        optimal_clusters = num_clusters
    else:
        optimal_clusters = find_optimal_clusters(results)
    final_birch = Birch(n_clusters=optimal_clusters)
    final_labels = final_birch.fit_predict(scaled_data)

    # Get centroids (BIRCH model does not return centroids directly, but we can use the subcluster centers)
    centroids = final_birch.subcluster_centers_
    if normalize:
        centroids = scaler.inverse_transform(centroids)

    if isinstance(data, pd.DataFrame):
        centroids = pd.DataFrame(centroids, columns=data.columns)

    # Create visualizations
    plot_birch_clustering_results(results, optimal_clusters)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return {
        "results_df": results_df,
        "optimal_clusters": optimal_clusters,
        "labels": final_labels,
        "final_model": final_birch,
        "centroids": centroids,
    }


def plot_birch_clustering_results(results, optimal_clusters):
    """
    Create visualization of clustering results.
    """
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Plot evaluation metrics with error bars
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])

    # Silhouette and Calinski-Harabasz scores
    ax1.errorbar(
        results["n_clusters"],
        results["silhouette_score"],
        yerr=results["silhouette_std"],
        fmt="bo-",
        label="Silhouette",
    )
    ax1_twin = ax1.twinx()
    ax1_twin.errorbar(
        results["n_clusters"],
        results["calinski_score"],
        yerr=results["calinski_std"],
        fmt="ro-",
        label="Calinski-Harabasz",
    )

    # Highlight optimal clusters
    ax1.axvline(
        x=optimal_clusters,
        color="g",
        linestyle="--",
        label=f"Optimal (n={optimal_clusters})",
    )

    ax1.set_title("Clustering Evaluation Metrics")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Silhouette Score", color="b")
    ax1_twin.set_ylabel("Calinski-Harabasz Score", color="r")

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Elbow plot (inertia)
    ax2.plot(results["n_clusters"], results["inertia"], "go-")
    ax2.axvline(x=optimal_clusters, color="r", linestyle="--")
    ax2.set_title("Elbow Plot (Inertia)")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Inertia (Sum of Squared Distances)")
    ax2.set_ylim(0, None)

    # Cross-validation stability plot
    ax3.fill_between(
        results["n_clusters"],
        np.array(results["silhouette_score"]) - np.array(results["silhouette_std"]),
        np.array(results["silhouette_score"]) + np.array(results["silhouette_std"]),
        alpha=0.2,
    )
    ax3.plot(results["n_clusters"], results["silhouette_score"], "b-")
    ax3.set_title("Cross-validation Stability")
    ax3.set_xlabel("Number of Clusters")
    ax3.set_ylabel("Silhouette Score with Std Dev")

    plt.tight_layout()
    plt.show()


def bisecting_kmeans_clustering_analysis(
    data,
    max_clusters=15,
    n_splits=3,
    random_state=42,
    num_clusters=None,
    normalize=False,
):
    """
    Perform Bisecting KMeans clustering analysis with cross-validation and automatic cluster selection.

    Parameters:
    data (DataFrame/array): Input data
    max_clusters (int): Maximum number of clusters to evaluate
    n_splits (int): Number of cross-validation splits
    random_state (int): Random state for reproducibility

    Returns:
    dict: Dictionary containing results, metrics, and optimal clusters
    """

    # Standardize the data (optional)
    if normalize:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.array(data))
    else:
        scaled_data = np.array(data)

    scaled_data = np.array(data)  # Assuming data is already scaled or numerical

    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize results dictionary
    results = {
        "n_clusters": [],
        "silhouette_score": [],
        "silhouette_std": [],
        "calinski_score": [],
        "calinski_std": [],
        "inertia": [],
    }

    # Evaluate different numbers of clusters with cross-validation
    for n in range(2, max_clusters + 1):
        silhouette_scores = []
        calinski_scores = []
        inertia_scores = []

        for train_idx, val_idx in kf.split(scaled_data):
            # Split data
            train_data = scaled_data[train_idx]
            val_data = scaled_data[val_idx]

            # Fit Bisecting KMeans clustering
            bisecting_kmeans = BisectingKMeans(n_clusters=n, random_state=random_state)
            bisecting_kmeans.fit(train_data)
            train_labels = bisecting_kmeans.labels_
            val_labels = bisecting_kmeans.predict(val_data)

            # Calculate metrics
            if len(np.unique(val_labels)) > 1:  # Check if more than one cluster
                silhouette_scores.append(silhouette_score(val_data, val_labels))
                calinski_scores.append(calinski_harabasz_score(val_data, val_labels))

            # Calculate inertia (sum of squared distances to cluster centers)
            inertia_scores.append(bisecting_kmeans.inertia_)

        # Store results
        results["n_clusters"].append(n)
        results["silhouette_score"].append(np.mean(silhouette_scores))
        results["silhouette_std"].append(np.std(silhouette_scores))
        results["calinski_score"].append(np.mean(calinski_scores))
        results["calinski_std"].append(np.std(calinski_scores))
        results["inertia"].append(np.mean(inertia_scores))

    # Find optimal number of clusters (using the elbow method on inertia plot)
    if num_clusters:
        optimal_clusters = num_clusters
    else:
        optimal_clusters = find_optimal_clusters(results)

    # Fit final Bisecting KMeans model with optimal clusters
    final_bisecting_kmeans = BisectingKMeans(
        n_clusters=optimal_clusters, random_state=random_state
    )
    final_labels = final_bisecting_kmeans.fit_predict(scaled_data)

    # Get centroids
    centroids = final_bisecting_kmeans.cluster_centers_
    if normalize:
        centroids = scaler.inverse_transform(centroids)

    if isinstance(data, pd.DataFrame):
        centroids = pd.DataFrame(centroids, columns=data.columns)

    plot_BKMeans_clustering_results(results, optimal_clusters)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return {
        "results_df": results_df,
        "optimal_clusters": optimal_clusters,
        "labels": final_labels,
        "final_model": final_bisecting_kmeans,
        "centroids": centroids,
    }


def plot_BKMeans_clustering_results(results, optimal_clusters):
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Plot evaluation metrics with error bars
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])

    # Silhouette and Calinski-Harabasz scores
    ax1.errorbar(
        results["n_clusters"],
        results["silhouette_score"],
        yerr=results["silhouette_std"],
        fmt="bo-",
        label="Silhouette",
    )
    ax1_twin = ax1.twinx()
    ax1_twin.errorbar(
        results["n_clusters"],
        results["calinski_score"],
        yerr=results["calinski_std"],
        fmt="ro-",
        label="Calinski-Harabasz",
    )

    # Highlight optimal clusters
    ax1.axvline(
        x=optimal_clusters,
        color="g",
        linestyle="--",
        label=f"Optimal (n={optimal_clusters})",
    )

    ax1.set_title("Clustering Evaluation Metrics")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Silhouette Score", color="b")
    ax1_twin.set_ylabel("Calinski-Harabasz Score", color="r")

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Elbow plot (inertia)
    ax2.plot(results["n_clusters"], results["inertia"], "go-")
    ax2.axvline(x=optimal_clusters, color="r", linestyle="--")
    ax2.set_title("Elbow Plot (Inertia)")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Inertia (Sum of Squared Distances)")
    ax2.set_ylim(0, None)

    # Cross-validation stability plot
    ax3.fill_between(
        results["n_clusters"],
        np.array(results["silhouette_score"]) - np.array(results["silhouette_std"]),
        np.array(results["silhouette_score"]) + np.array(results["silhouette_std"]),
        alpha=0.2,
    )
    ax3.plot(results["n_clusters"], results["silhouette_score"], "b-")
    ax3.set_title("Cross-validation Stability")
    ax3.set_xlabel("Number of Clusters")
    ax3.set_ylabel("Silhouette Score with Std Dev")

    plt.tight_layout()
    plt.show()


def find_optimal_clusters(results):
    """
    Find optimal number of clusters using multiple metrics and proper normalization.
    """

    n_clusters = np.array(results["n_clusters"])

    # Normalize metrics between 0 and 1
    silhouette = np.array(results["silhouette_score"])
    silhouette_norm = (silhouette - silhouette.min()) / (
        silhouette.max() - silhouette.min()
    )

    calinski = np.array(results["calinski_score"])
    calinski_norm = (calinski - calinski.min()) / (calinski.max() - calinski.min())

    # Calculate elbow score using acceleration method
    inertia = np.array(results["inertia"])
    acceleration = np.diff(np.diff(inertia))
    k_elbow = n_clusters[2:][np.argmax(acceleration)]

    # Create elbow score array
    elbow_score = np.zeros_like(silhouette)
    for i, k in enumerate(n_clusters):
        elbow_score[i] = 1 / (1 + np.abs(k - k_elbow))

    # Weighted combination of metrics
    weights = {"silhouette": 0, "calinski": 0, "elbow": 1}

    combined_score = (
        weights["silhouette"] * silhouette_norm
        + weights["calinski"] * calinski_norm
        + weights["elbow"] * elbow_score
    )

    optimal_idx = np.argmax(combined_score)
    return n_clusters[optimal_idx]


def get_silhouette_score(results, num_clusters):
    results_df = results["results_df"]
    silhouette_score = results_df["silhouette_score"].iloc[num_clusters - 2]
    silhouette_std = results_df["silhouette_std"].iloc[num_clusters - 2]
    calinski_score = results_df["calinski_score"].iloc[num_clusters - 2]
    calinski_std = results_df["calinski_std"].iloc[num_clusters - 2]

    if "inertia" in results_df.columns:
        inertia = results_df["inertia"].iloc[num_clusters - 2]
        print(f"Inertia: {inertia:.3f}")
    print(f"Silhouette score: {silhouette_score:.3f} +/- {silhouette_std:.3f}")
    print(f"Calinski-Harabasz score: {calinski_score:.3f} +/- {calinski_std:.3f}")
    # print(f'Inertia: {inertia:.3f}')
    if "inertia" in results_df.columns:
        print(
            f"{num_clusters}, {silhouette_score:.2f} + {silhouette_std:.2f}, {calinski_score:.2f} + {calinski_std:.2f}, {inertia:.2f}"
        )
    else:
        print(
            f"{num_clusters}, {silhouette_score:.2f} + {silhouette_std:.2f}, {calinski_score:.2f} + {calinski_std:.2f}"
        )


# Function to perform clustering and return results
def perform_clustering(clustering_method, cluster_df, random_state, num_clusters):
    method_mapping = {
        "KMeans": kmeans_clustering_analysis,  # 71
        "BisectKMean": bisecting_kmeans_clustering_analysis,  # 77
        "hierachical": hierarchical_clustering_analysis,  # 22
        "affinity_propagation": affinity_propagation_analysis,  # 85
        "GaussianMixture": gaussian_mixture_clustering_analysis,  # 77
        "Birch": birch_clustering_analysis,  # 78
    }

    # Perform clustering analysis based on the selected method
    clustering_func = method_mapping.get(clustering_method)

    if clustering_method == "affinity_propagation":
        results = clustering_func(
            cluster_df,
            n_splits=3,
            random_state=random_state,
            num_clusters=7,
            normalize=False,
        )
        labels, centers, exemplars = get_ap_cluster_labels(
            cluster_df, preference=-10, damping=0.65, random_state=random_state
        )
    else:
        results = clustering_func(
            cluster_df,
            n_splits=3,
            random_state=random_state,
            num_clusters=num_clusters,
            normalize=False,
        )
        labels = results["labels"]
        centroids = results["centroids"]

    num_clusters = len(np.unique(labels))
    print(
        f"Optimal number of clusters: {results.get('optimal_clusters', num_clusters)}"
    )
    get_silhouette_score(results, num_clusters)

    return labels, centroids, num_clusters
