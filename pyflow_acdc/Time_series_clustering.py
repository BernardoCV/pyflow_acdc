from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering, HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as time
from pathlib import Path
from scipy.spatial.distance import cdist



__all__ = ['cluster_TS',
           'cluster_Kmeans',
           'cluster_Ward',
           'cluster_DBSCAN',
           'cluster_OPTICS',
           'cluster_Kmedoids',
           'cluster_Spectral',
           'cluster_HDBSCAN',
           'run_clustering_analysis_and_plot',
           'identify_correlations']

def get_cluster_sizes(data):
    """
    Helper function to calculate cluster sizes.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with 'Cluster' column containing cluster assignments
    
    Returns:
    --------
    list
        Number of points in each cluster, sorted by cluster index
    """
    if 'Cluster' not in data.columns:
        raise ValueError("Data must contain a 'Cluster' column")
        
    cluster_counts = data['Cluster'].value_counts().sort_index()
    sizes = cluster_counts.values.tolist()
    
    # Print warning if any clusters are empty
    n_clusters = len(set(data['Cluster']))
    if len(sizes) != n_clusters:
        print(f"Warning: Some clusters are empty. Found {len(sizes)} non-empty clusters out of {n_clusters}")
    
    return sizes

def filter_data(grid,time_series):
    data = pd.DataFrame()
    for ts in grid.Time_series:
        if ts.type in time_series:
            name = ts.name
            ts_data = ts.data
            if data.empty:
                data[name] = ts_data
                expected_length = len(ts_data)
            else:
                # Check if ts_data length matches the expected length
                if len(ts_data) != expected_length:
                    print(f"Error: Length mismatch for time series '{name}'. Expected {expected_length}, got {len(ts_data)}. Time series not included")
                    continue
                data[name] = ts_data        
    return data

def identify_correlations(grid,time_series=[], threshold=0.8):
    """
    Identify highly correlated time series variables.
    
    Parameters:
        grid: Grid object containing time series
        threshold: Correlation coefficient threshold (default: 0.8)
    
    Returns:
        dict: Dictionary containing:
            - correlation_matrix: Full correlation matrix
            - high_correlations: List of tuples (var1, var2, corr_value) for highly correlated pairs
            - groups: List of groups of correlated variables
    """
    # Create DataFrame from time series
    if time_series == []:
        time_series = [
                'a_CG',     # Price zone cost generation parameter a
                'b_CG',     # Price zone cost generation parameter b
                'c_CG',     # Price zone cost generation parameter c
                'PGL_min',  # Price zone minimum generation limit
                'PGL_max',  # Price zone maximum generation limit
                'price',    # Price for price zones and AC nodes
                'Load',     # Load factor for price zones and AC nodes
                'WPP',      # Wind Power Plant availability
                'OWPP',     # Offshore Wind Power Plant availability
                'SF',       # Solar Farm availability
                'REN'       # Generic Renewable source availability
            ]
    data = filter_data(grid,time_series)
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Find highly correlated pairs
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > threshold:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                high_corr.append((var1, var2, corr))
    
    # Group correlated variables
    groups = []
    used_vars = set()
    
    for var1, var2, corr in high_corr:
        # Find if any existing group contains either variable
        found_group = False
        for group in groups:
            if var1 in group or var2 in group:
                group.add(var1)
                group.add(var2)
                found_group = True
                break
        
        # If no existing group found, create new group
        if not found_group:
            groups.append({var1, var2})
        
        used_vars.add(var1)
        used_vars.add(var2)
    
    # Print results
    print(f"\nHighly correlated variables (|correlation| > {threshold}):")
    for var1, var2, corr in high_corr:
        print(f"{var1:20} - {var2:20}: {corr:.3f}")
    
    print("\nCorrelated groups:")
    for i, group in enumerate(groups, 1):
        print(f"Group {i}: {', '.join(sorted(group))}")
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_corr,
        'groups': groups
    }

def plot_correlation_matrix(corr_matrix, save_path=None):
    """
    Plot correlation matrix as a heatmap.
    
    Parameters:
        corr_matrix: Pandas DataFrame with correlation matrix
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    plt.imshow(corr_matrix, cmap='RdBu', aspect='equal', vmin=-1, vmax=1)
    
    # Add labels
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center')
    
    plt.title('Correlation Matrix of Time Series')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def cluster_TS(grid, n_clusters,time_series=[], algorithm='Kmeans'):
    algorithm = algorithm.lower()
    #check if algorithm is valid    
    if algorithm not in {'kmeans','ward','dbscan','optics','kmedoids','spectral','hdbscan'}:
        algorithm='kmeans'
        print(f"Algorithm {algorithm} not found, using Kmeans")
    
    #create data from grid
    if time_series == []:
        time_series = [
                'a_CG',     # Price zone cost generation parameter a
                'b_CG',     # Price zone cost generation parameter b
                'c_CG',     # Price zone cost generation parameter c
                'PGL_min',  # Price zone minimum generation limit
                'PGL_max',  # Price zone maximum generation limit
                'price',    # Price for price zones and AC nodes
                'Load',     # Load factor for price zones and AC nodes
                'WPP',      # Wind Power Plant availability
                'OWPP',     # Offshore Wind Power Plant availability
                'SF',       # Solar Farm availability
                'REN'       # Generic Renewable source availability
            ]
    data = filter_data(grid,time_series)
    
    if algorithm == 'kmeans':
        clusters, returns, labels = cluster_Kmeans(grid, n_clusters, data,ts_types=time_series)
    elif algorithm == 'ward':
        clusters, returns, labels = cluster_Ward(grid, n_clusters, data,ts_types=time_series)
    elif algorithm == 'kmedoids':
        clusters, returns, labels = cluster_Kmedoids(grid, n_clusters, data,ts_types=time_series)
    elif algorithm == 'spectral':
        clusters, returns, labels = cluster_Spectral(grid, n_clusters, data,ts_types=time_series)
    elif algorithm == 'dbscan':
        n_clusters, clusters, returns, labels = cluster_DBSCAN(grid, n_clusters, data,ts_types=time_series)
    elif algorithm == 'optics':
        n_clusters, clusters, returns, labels = cluster_OPTICS(grid, n_clusters, data,ts_types=time_series)    
    elif algorithm == 'hdbscan':
        n_clusters, clusters, returns, labels = cluster_HDBSCAN(grid, n_clusters, data,ts_types=time_series)
    

    return n_clusters, clusters, returns, labels

def _process_clusters(grid, data, cluster_centers, n_clusters, new_columns,ts_types):
    """
    Process clustering results and update grid with cluster information.
    
    Parameters:
    -----------
    grid : Grid object
        The grid object to update
    data : pandas.DataFrame
        Data with cluster assignments
    cluster_centers : numpy.ndarray
        Centers of the clusters
    n_clusters : int
        Number of clusters
    new_columns : list
        Column names for the cluster centers DataFrame
    """
    # Create DataFrame with cluster centers
    clusters = pd.DataFrame(cluster_centers, columns=new_columns)
    
    # Calculate cluster counts and weights
    cluster_counts = data['Cluster'].value_counts().sort_index()
    total_count = len(data)
    cluster_weights = cluster_counts / total_count
    
    # Add counts and weights to clusters DataFrame
    clusters.insert(0, 'Cluster Count', cluster_counts.values)
    clusters.insert(1, 'Weight', cluster_weights.values)
    
    # Update grid with cluster weights
    grid.Clusters[n_clusters] = clusters['Weight'].to_numpy(dtype=float)
    
    # Update time series with clustered data
    for ts in grid.Time_series:
        if ts.type in ts_types:
            if not hasattr(ts, 'data_clustered') or not isinstance(ts.data_clustered, dict):
                ts.data_clustered = {}
            name = ts.name
            ts.data_clustered[n_clusters] = clusters[name].to_numpy(dtype=float)
    
    return clusters

def cluster_OPTICS(grid, n_clusters, data, min_samples=2, max_eps=np.inf, xi=0.05,ts_types=[]):
    """
    Perform OPTICS clustering on the data with maximum number of clusters constraint.
    
    Parameters:
    -----------
    grid : Grid object
        The grid object to update
    n_clusters : int
        Maximum number of clusters desired
    data : pandas.DataFrame
        Data to cluster
    min_samples : int, default=2
        The number of samples in a neighborhood for a point to be considered a core point
    max_eps : float, default=np.inf
        The maximum distance between two samples
    xi : float, default=0.05
        Determines the minimum steepness on the reachability plot
    """
    new_columns = data.columns
    
    # Scale the data for clustering
   
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    
    # Try different xi values until we get desired number of clusters
    best_labels = None
    best_xi = None
    current_xi = xi
    
    while current_xi <= 1.0:  # xi must be between 0 and 1
        optics = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=current_xi)
        labels = optics.fit_predict(data_scaled)
        
        # Count actual clusters (excluding noise)
        actual_clusters = len(set(labels[labels >= 0]))
        
        if actual_clusters <= n_clusters and actual_clusters > 0:
            best_labels = labels
            best_xi = current_xi
            break
        elif actual_clusters > n_clusters:
            current_xi *= 1.5  # Increase xi to get fewer clusters
        else:  # No clusters found
            current_xi *= 0.8  # Decrease xi to get more clusters
    
    if best_labels is None:
        print("Warning: Could not find suitable clustering. Try adjusting parameters.")
        return 0, None
    
    # Use best result
    data['Cluster'] = best_labels
    labels = data['Cluster']
    actual_clusters = len(set(best_labels[best_labels >= 0]))
    
    # Calculate cluster centers in scaled space
    cluster_centers_scaled = []
    unique_clusters = sorted(set(data['Cluster']))
    for cluster_id in unique_clusters:
        cluster_data = data_scaled[data['Cluster'] == cluster_id]
        cluster_centers_scaled.append(cluster_data.mean().values)
    cluster_centers_scaled = np.array(cluster_centers_scaled)
    
    # Transform cluster centers back to original scale
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    
    # Print clustering results
    cluster_sizes = get_cluster_sizes(data)
    noise_points = len(data[data['Cluster'] == -1])
    noise_percentage = (noise_points / len(data)) * 100
    
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Found clusters": actual_clusters,
        "Maximum allowed": n_clusters,
        "Final xi": best_xi,
        "Noise points": (noise_points, noise_percentage)
    }
    CoV = print_clustering_results("OPTICS", actual_clusters, specific_info)
    
    # Process and return results
    processed_results = _process_clusters(grid, data, cluster_centers, actual_clusters, new_columns,ts_types)
    return actual_clusters, processed_results, CoV, [data_scaled,labels]


def cluster_DBSCAN(grid, n_clusters, data, min_samples=2, initial_eps=0.5,ts_types=[]):
    """
    Perform DBSCAN clustering on the data with maximum number of clusters.
    """
    new_columns = data.columns
    
    # Scale the data for clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    
    eps = initial_eps
    best_result = None
    best_eps = None
    
    # Try different eps values until we find clusters
    while eps <= 10.0:  # Set a reasonable maximum eps
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_scaled)
        
        # Get actual number of clusters (excluding noise points marked as -1)
        actual_clusters = len(set(labels[labels >= 0]))
        
        if actual_clusters > 0:  # If we found any clusters
            if actual_clusters <= n_clusters:
                best_result = labels
                best_eps = eps
                break
            else:
                eps *= 1.1  # Try larger eps for fewer clusters
        else:
            eps *= 1.5  # Significantly increase eps if no clusters found
        
    if best_result is None:
        print("Warning: Could not find any meaningful clusters. Try adjusting parameters.")
        return 0, None
    
    # Use best result
    data['Cluster'] = best_result
    labels = data['Cluster']
    actual_clusters = len(set(best_result[best_result >= 0]))
    
    # Calculate cluster centers in scaled space
    cluster_centers_scaled = []
    unique_clusters = sorted(set(data['Cluster']))
    for cluster_id in unique_clusters:
        cluster_data = data_scaled[data['Cluster'] == cluster_id]
        cluster_centers_scaled.append(cluster_data.mean().values)
    cluster_centers_scaled = np.array(cluster_centers_scaled)
    
    # Transform cluster centers back to original scale
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    
    # Print clustering results
    cluster_sizes = get_cluster_sizes(data)
    noise_points = len(data[data['Cluster'] == -1])
    noise_percentage = (noise_points / len(data)) * 100
    
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Found clusters": actual_clusters,
        "Maximum allowed": n_clusters,
        "Final eps": best_eps,
        "Noise points": (noise_points, noise_percentage)
    }
    CoV = print_clustering_results("DBSCAN", actual_clusters, specific_info)
    
    # Always call _process_clusters with valid results
    processed_results = _process_clusters(grid, data, cluster_centers, actual_clusters, new_columns,ts_types)
    return actual_clusters, processed_results, CoV, [data_scaled,labels]

def cluster_Ward(grid, n_clusters, data,ts_types=[]):
    """
    Perform Ward's hierarchical clustering using AgglomerativeClustering.
    
    Parameters:
    -----------
    grid : Grid object
        The grid object to update
    n_clusters : int
        Number of clusters
    data : pandas.DataFrame
        Data to cluster
    """
    new_columns = data.columns
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    
    # Perform Ward's hierarchical clustering on scaled dat
    ward = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward',
        compute_distances=True  # Enables distance computation
    )
    data['Cluster'] = ward.fit_predict(data_scaled)
    labels = data['Cluster']
    # Calculate cluster centers in scaled space
    cluster_centers_scaled = []
    cluster_sizes = get_cluster_sizes(data)
    
    for cluster_id in range(n_clusters):
        cluster_data = data_scaled[data['Cluster'] == cluster_id]
        cluster_centers_scaled.append(cluster_data.mean(axis=0))
    cluster_centers_scaled = np.array(cluster_centers_scaled)
    
    # Transform centers back to original scale
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    
    # Get additional metrics
    distances = ward.distances_  # Available if compute_distances=True
    
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Maximum merge distance": float(max(distances)) if len(distances) > 0 else 0,
        "Average merge distance": float(np.mean(distances)) if len(distances) > 0 else 0
    }
    CoV = print_clustering_results("Ward hierarchical", n_clusters, specific_info)
    
    processed_results = _process_clusters(grid, data, cluster_centers, n_clusters, new_columns,ts_types)
    return  processed_results, CoV, [data_scaled,labels]

def cluster_Kmeans(grid, n_clusters, data,ts_types=[]):
    new_columns = data.columns
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    
    # Fit KMeans on scaled data
    kmeans = KMeans(n_clusters=n_clusters)
    data['Cluster'] = kmeans.fit_predict(data_scaled)
    labels = data['Cluster']
    
    # Get cluster centers and transform back to original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Print clustering results
    cluster_sizes = get_cluster_sizes(data)
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Inertia": kmeans.inertia_,
        "Iterations": kmeans.n_iter_
    }
    CoV = print_clustering_results("K-means", n_clusters, specific_info)
    
    processed_results = _process_clusters(grid, data, cluster_centers, n_clusters, new_columns,ts_types)
    return  processed_results, [CoV,kmeans.inertia_,kmeans.n_iter_], [data_scaled,labels]

def cluster_Kmedoids(grid, n_clusters, data, method='alternate', init='build', max_iter=300,ts_types=[]):
    """
    Perform K-Medoids clustering on the data.
    
    Parameters:
    -----------
    grid : Grid object
        The grid object to update
    n_clusters : int
        Number of clusters
    data : pandas.DataFrame
        Data to cluster
    method : str, default='alternate'
        {'alternate', 'pam'} Algorithm to use
    init : str, default='build'
        {'random', 'heuristic', 'k-medoids++'} Initialization method
    max_iter : int, default=300
        Maximum number of iterations
    """
    new_columns = data.columns
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    
    # Import KMedoids
    
    
    # Fit KMedoids on scaled data
    kmedoids = KMedoids(
        n_clusters=n_clusters,
        method=method,
        init=init,
        max_iter=max_iter
    )
    data['Cluster'] = kmedoids.fit_predict(data_scaled)
    labels = data['Cluster']

    # Get medoid indices
    medoid_indices = kmedoids.medoid_indices_
    
    # Get cluster centers (medoids) in original scale
    cluster_centers = data.iloc[medoid_indices, :-1].values  # Exclude 'Cluster' column
    
    # Print clustering results
    cluster_sizes = get_cluster_sizes(data)
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Method": method,
        "Initialization": init,
        "Inertia": kmedoids.inertia_
    }
    CoV = print_clustering_results("K-medoids", n_clusters, specific_info)
    
    processed_results = _process_clusters(grid, data, cluster_centers, n_clusters, new_columns,ts_types)
    return  processed_results, [CoV,kmedoids.inertia_], [data_scaled,labels]

def cluster_Spectral(grid, n_clusters, data, n_init=10, assign_labels='kmeans', affinity='rbf', gamma=1.0,ts_types=[]):
    """
    Perform Spectral clustering on the data.
    
    Parameters:
    -----------
    grid : Grid object
        The grid object to update
    n_clusters : int
        Number of clusters
    data : pandas.DataFrame
        Data to cluster
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different centroid seeds
    assign_labels : {'kmeans', 'discretize'}, default='kmeans'
        Strategy to assign labels in the embedding space
    affinity : {'rbf', 'nearest_neighbors', 'precomputed'}, default='rbf'
        How to construct the affinity matrix
    gamma : float, default=1.0
        Kernel coefficient for rbf kernel
    """
    new_columns = data.columns
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        n_init=n_init,
        assign_labels=assign_labels,
        affinity=affinity,
        gamma=gamma,
        random_state=42
    )
    
    # Fit and predict
    data['Cluster'] = spectral.fit_predict(data_scaled)
    labels = data['Cluster']
    # Calculate cluster centers in scaled space
    cluster_centers_scaled = []
    cluster_sizes = get_cluster_sizes(data)
    
    for cluster_id in range(n_clusters):
        cluster_data = data_scaled[data['Cluster'] == cluster_id]
        cluster_centers_scaled.append(cluster_data.mean(axis=0))
    cluster_centers_scaled = np.array(cluster_centers_scaled)
    
    # Transform centers back to original scale
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    
    # Get affinity matrix properties
    affinity_matrix = spectral.affinity_matrix_
    connectivity = (affinity_matrix > 0).sum() / (affinity_matrix.shape[0] * affinity_matrix.shape[1])
    
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Affinity": affinity,
        "Label assignment": assign_labels,
        "Gamma": gamma,
        "Connectivity density": f"{connectivity:.2%}",
        "Average affinity": f"{affinity_matrix.mean():.4f}"
    }
    CoV = print_clustering_results("Spectral", n_clusters, specific_info)
    
    processed_results = _process_clusters(grid, data, cluster_centers, n_clusters, new_columns,ts_types)
    return  processed_results, CoV, [data_scaled,labels]

def cluster_HDBSCAN(grid, n_clusters, data, min_cluster_size=5, min_samples=None, cluster_selection_method='eom',ts_types=[]):
    """
    Perform HDBSCAN clustering on the data.
    
    Parameters:
    -----------
    grid : Grid object
        The grid object to update
    n_clusters : int
        Soft constraint on number of clusters (HDBSCAN determines optimal number)
    data : pandas.DataFrame
        Data to cluster
    min_cluster_size : int, default=5
        The minimum size of clusters
    min_samples : int, default=None
        The number of samples in a neighborhood for a point to be a core point
    cluster_selection_method : {'eom', 'leaf'}, default='eom'
        The method used to select clusters
    """
    new_columns = data.columns
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    
    # If min_samples not specified, use min_cluster_size
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Initialize HDBSCA
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method
    )
    
    # Fit and predict
    data['Cluster'] = clusterer.fit_predict(data_scaled)
    labels = data['Cluster']
    # Get actual number of clusters (excluding noise points marked as -1)
    actual_clusters = len(set(data['Cluster'][data['Cluster'] >= 0]))
    
    # Calculate cluster centers in scaled space
    cluster_centers_scaled = []
    unique_clusters = sorted(set(data['Cluster']))
    for cluster_id in unique_clusters:
        cluster_data = data_scaled[data['Cluster'] == cluster_id]
        cluster_centers_scaled.append(cluster_data.mean().values)
    cluster_centers_scaled = np.array(cluster_centers_scaled)
    
    # Transform centers back to original scale
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    
    # Get cluster sizes and noise points
    cluster_sizes = get_cluster_sizes(data)
    noise_points = len(data[data['Cluster'] == -1])
    noise_percentage = (noise_points / len(data)) * 100
    
    specific_info = {
        "Found clusters": actual_clusters,
        "Target clusters": n_clusters,
        "Cluster sizes": cluster_sizes,
        "Noise points": (noise_points, noise_percentage),
        "Min cluster size": min_cluster_size,
        "Min samples": min_samples,
        "Selection method": cluster_selection_method,
        "Probabilities available": hasattr(clusterer, 'probabilities_')
    }
    CoV = print_clustering_results("HDBSCAN", actual_clusters, specific_info)
    
    processed_results = _process_clusters(grid, data, cluster_centers, actual_clusters, new_columns,ts_types)
    return actual_clusters, processed_results , CoV, [data_scaled,labels]


def dunn_index(X, labels):
    """
    Compute the Dunn Index for clustering results.

    Parameters:
        X (array-like): Data points (n_samples, n_features).
        labels (array-like): Cluster labels for each data point.

    Returns:
        float: Dunn Index value (higher is better).
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    if num_clusters < 2:
        return 0  # Dunn Index is undefined for a single cluster

    # Compute intra-cluster distances (max within-cluster distance)
    intra_dists = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            intra_dists.append(np.max(pairwise_distances(cluster_points)))
        else:
            intra_dists.append(0)  # Single-point cluster

    max_intra_dist = np.max(intra_dists) if intra_dists else 0

    # Compute inter-cluster distances (min distance between different clusters)
    inter_dists = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            dist_matrix = cdist(cluster_i, cluster_j)  # Compute distances between clusters
            inter_dists.append(np.min(dist_matrix))

    min_inter_dist = np.min(inter_dists) if inter_dists else 0

    if max_intra_dist == 0:
        return 0
    
    return min_inter_dist / max_intra_dist

def print_clustering_results(algorithm, n_clusters, specific_info):
    """Helper function to print clustering results in a standardized format."""
    print(f"\n{algorithm} clustering results:")
    print(f"- Number of clusters: {n_clusters}")
    CoV=0
    # Print algorithm-specific information
    for key, value in specific_info.items():
        if isinstance(value, (int, str)):
            print(f"- {key}: {value}")
        elif isinstance(value, float):
            print(f"- {key}: {value:.2f}")
        elif isinstance(value, list):
            print(f"- {key}: {value}")
            
            if key == "Cluster sizes":
                CoV = np.std(value)/np.mean(value)
                print(f"  • Average: {np.mean(value):.1f}")
                print(f"  • Std dev: {np.std(value):.1f}")
                print(f"  • CoV    : {CoV:.1f}")
        elif isinstance(value, tuple):
            count, percentage = value
            print(f"- {key}: {count} ({percentage:.1f}%)")
    return CoV    

def run_clustering_analysis(grid, save_path='clustering_results',algorithms = ['kmeans', 'kmedoids', 'ward', 'dbscan', 'hdbscan'],n_clusters_list = [1, 4, 8, 16, 24, 48],time_series=[]):
       
    
    results = {
        'algorithm': [],
        'n_clusters': [],
        'time_taken': [],
        'Coefficient of Variation': [],
        'inertia': [],
        'silhouette_score': [],
        'dunn_index': [],
        'davies_bouldin': []
    }
    
    for algo in algorithms:
        print(f"\nTesting {algo}...")
        for n in n_clusters_list:
            print(f"  Clusters: {n}")
            
            start_time = time.time()
            try:
                _,_,CoV,info = cluster_TS(grid, algorithm=algo, n_clusters=n,time_series=time_series)
                data_scaled,labels = info
                if algo == 'kmeans':
                    CoV, inertia, n_iter_ = CoV
                elif algo == 'kmedoids':
                    CoV, inertia = CoV
                else:
                    inertia = 0
                time_taken = time.time() - start_time

                if labels is not None and len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(data_scaled,labels)
                    dunn_idx = dunn_index(data_scaled,labels)
                    db_score = davies_bouldin_score(data_scaled,labels)
                else:
                    sil_score = dunn_idx = db_score = 0

                results['algorithm'].append(algo)
                results['n_clusters'].append(n)
                results['time_taken'].append(time_taken)
                results['Coefficient of Variation'].append(CoV)
                results['inertia'].append(inertia)
                results['silhouette_score'].append(sil_score)
                results['dunn_index'].append(dunn_idx)
                results['davies_bouldin'].append(db_score)
                
                print(f"    Time: {time_taken:.2f}s")
                
            except Exception as e:
                print(f"    Error with {algo}, n={n}: {str(e)}")
                continue
    
    df_results = pd.DataFrame(results)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Updated summary to use correct columns
    summary_df = df_results[['algorithm', 'n_clusters', 'time_taken', 'Coefficient of Variation','inertia','silhouette_score','dunn_index','davies_bouldin']]
    summary_df.to_csv(f'{save_path}/clustering_summary.csv', index=False)
 
    
    return df_results

# Usage:
# results = run_clustering_analysis(grid)

# To analyze results:
def plot_clustering_results(df= None,results_path='clustering_results',format='svg'):
    # Convert 8.25 cm to inches and maintain ratio
    width_cm = 8.25
    ratio = 6/10  # Original height/width ratio
    width_inches = width_cm / 2.54
    height_inches = width_inches * ratio
    
    # Set global plotting parameters
    plt.rcParams.update({
        'figure.figsize': (width_inches, height_inches),
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.markersize': 4,
        'lines.linewidth': 1
    })
    
    if df is None:
        df = pd.read_csv(f'{results_path}/clustering_summary.csv')
    
    def format_axes(ax):
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 1. Time comparison plot
    plt.figure()
    ax = plt.gca()
    for algo in df['algorithm'].unique():
        data = df[df['algorithm'] == algo]
        ax.plot(data['n_clusters'], data['time_taken'], 
                marker='o', label=algo)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Time (seconds)')
    ax.legend()
    format_axes(ax)
    plt.tight_layout()
    plt.savefig(f'{results_path}/time_comparison.{format}', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Standard deviation plot
    fig, ax = plt.subplots()
    for algo in df['algorithm'].unique():
        data = df[df['algorithm'] == algo]
        ax.plot(data['n_clusters'], data['Coefficient of Variation'], 
                marker='o', label=algo)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Coefficient of Variation')
    ax.legend()
    format_axes(ax)
    plt.tight_layout()
    plt.savefig(f'{results_path}/cov_comparison.{format}', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Inertia plot
    fig, ax = plt.subplots()
    kmeans_data = df[df['algorithm'] == 'kmeans']
    kmedoids_data = df[df['algorithm'] == 'kmedoids']
    
    ax.plot(kmeans_data['n_clusters'], kmeans_data['inertia'], 
            marker='o', label='k-means', linestyle='-')
    ax.plot(kmedoids_data['n_clusters'], kmedoids_data['inertia'], 
            marker='s', label='k-medoids', linestyle='-')
    
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.legend()
    format_axes(ax)
    plt.tight_layout()
    plt.savefig(f'{results_path}/inertia_comparison.{format}', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Silhouette score plot
    fig, ax = plt.subplots()
    for algo in df['algorithm'].unique():
        data = df[df['algorithm'] == algo]
        ax.plot(data['n_clusters'], data['silhouette_score'], 
                marker='o', label=algo)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.legend()
    format_axes(ax)
    plt.tight_layout()
    plt.savefig(f'{results_path}/silhouette_comparison.{format}', dpi=300, bbox_inches='tight')
    plt.close()     
    
    # 5. Dunn index plot
    fig, ax = plt.subplots()
    for algo in df['algorithm'].unique():
        data = df[df['algorithm'] == algo]
        ax.plot(data['n_clusters'], data['dunn_index'], 
                marker='o', label=algo)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Dunn Index')
    ax.legend()
    format_axes(ax)
    plt.tight_layout()
    plt.savefig(f'{results_path}/dunn_index_comparison.{format}', dpi=300, bbox_inches='tight')
    plt.close() 
    
    # 6. Davies-Bouldin index plot
    fig, ax = plt.subplots()
    for algo in df['algorithm'].unique():
        data = df[df['algorithm'] == algo]  
        ax.plot(data['n_clusters'], data['davies_bouldin'], 
                marker='o', label=algo)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.legend()
    format_axes(ax)
    plt.tight_layout()
    plt.savefig(f'{results_path}/davies_bouldin_comparison.{format}', dpi=300, bbox_inches='tight')
    plt.close()
    
  

def run_clustering_analysis_and_plot(grid,algorithms = ['kmeans', 'kmedoids', 'ward', 'dbscan', 'hdbscan'],n_clusters_list = [1, 4, 8, 16, 24, 48],path='clustering_results',time_series=[],plot_format='svg'):
    results = run_clustering_analysis(grid,path,algorithms,n_clusters_list,time_series)
    plot_clustering_results(results,path,format=plot_format)

def Time_series_cluster_relationship(grid, ts1_name=None, ts2_name=None,price_zone=None,ts_type=None, algorithm='kmeans', 
                            take_into_account_time_series=[], 
                            number_of_clusters=2, path='clustering_results', 
                            format='svg'):
    """
    Plot two time series with their cluster assignments in different colors.
    """
    # Get clusters
    n_clusters, clusters, returns, labels = cluster_TS(
        grid, number_of_clusters,time_series=take_into_account_time_series, algorithm=algorithm)
    data_scaled,labels = labels

    if ts1_name is not None:    
        ts1 = grid.Time_series[grid.Time_series_dic[ts1_name]].data
        if ts2_name is not None:
            ts2 = grid.Time_series[grid.Time_series_dic[ts2_name]].data
            plot_clustered_timeseries_single(ts1,ts2,algorithm,n_clusters,path,labels,ts1_name,ts2_name)
            return
        else:
            for ts in grid.Time_series.values():
                if ts.name != ts1_name:
                    ts2 = ts.data
                    ts2_name = ts.name
                    plot_clustered_timeseries_single(ts1,ts2,algorithm,n_clusters,path,labels,ts1_name,ts2_name)
            return
    elif price_zone is not None:
        PZ = grid.Price_Zones_dic[price_zone]
        # Collect all time series in a list
        ts_list = []
        ts_names = []
        for ts_idx in grid.Price_Zones[PZ].TS_dict.values():
            if ts_idx is None:
                continue
            ts = grid.Time_series[ts_idx]
            
            ts_list.append(ts.data)
            ts_names.append(ts.name)
        
        # Create plots for all pairs
        for i, ts1 in enumerate(ts_list):
            for j, ts2 in enumerate(ts_list[i+1:], start=i+1):
                plot_clustered_timeseries_single(
                    ts1=ts1,
                    ts2=ts2,
                    algorithm=algorithm,
                    n_clusters=n_clusters,
                    path=path,
                    labels=labels,
                    ts1_name=ts_names[i],
                    ts2_name=ts_names[j]
                )
    elif ts_type is not None:
        # Collect all time series of the specified type
        ts_list = []
        ts_names = []
        for ts in grid.Time_series:
            if ts.type == ts_type:
                ts_list.append(ts.data)
                ts_names.append(ts.name)
        
        # Create plots for all pairs
        for i, ts1 in enumerate(ts_list):
            for j, ts2 in enumerate(ts_list[i+1:], start=i+1):
                plot_clustered_timeseries_single(
                    ts1=ts1,
                    ts2=ts2,
                    algorithm=algorithm,
                    n_clusters=n_clusters,
                    path=path,
                    labels=labels,
                    ts1_name=ts_names[i],
                    ts2_name=ts_names[j]
                )
    else:
        print('No valid input provided')

def plot_clustered_timeseries_single(ts1,ts2,algorithm,n_clusters,path,labels,ts1_name,ts2_name): 
    # Get the time series data
    # Set up figure dimensions
    width_cm = 8.25
    width_inches = width_cm / 2.54
    height_inches = width_inches 
    
    # Set global plotting parameters
    plt.rcParams.update({
        'figure.figsize': (width_inches, height_inches),
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.markersize': 4,
        'lines.linewidth': 1
    })
    
    # Create color map for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot time series relationship
    plt.figure()
    for i in range(n_clusters):
        mask = labels == i
        plt.plot(ts1[mask], ts2[mask], 'o', 
                color=colors[i], label=f'Cluster {i}')
    plt.xlabel(ts1_name)
    plt.ylabel(ts2_name)
    plt.legend()
    plt.savefig(f'{path}/clustered_relationship_{algorithm}_{n_clusters}.png')
    plt.close()