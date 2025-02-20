from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering, HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
import pandas as pd
import numpy as np


__all__ = ['cluster_TS',
           'cluster_Kmeans',
           'cluster_Ward',
           'cluster_DBSCAN',
           'cluster_OPTICS',
           'cluster_Kmedoids',
           'cluster_Spectral',
           'cluster_HDBSCAN']

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

def cluster_TS(grid,n_clusters,algorithm ='Kmeans'):

    algorithm = algorithm.lower()
    #check if algorithm is valid    
    if algorithm not in {'kmeans','ward','dbscan','optics','kmedoids','spectral','hdbscan'}:
        algorithm='kmeans'
        print(f"Algorithm {algorithm} not found, using Kmeans")
    #create data from grid
    data = pd.DataFrame()
    
    for ts in grid.Time_series:
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
  
    
    if algorithm == 'kmeans':
        clusters = cluster_Kmeans(grid,n_clusters,data)
    elif algorithm == 'ward':
        clusters = cluster_Ward(grid,n_clusters,data)
    elif algorithm == 'dbscan':
        n_clusters, clusters = cluster_DBSCAN(grid,n_clusters,data)
    elif algorithm == 'optics':
        n_clusters, clusters = cluster_OPTICS(grid,n_clusters,data)
    elif algorithm == 'kmedoids':
        clusters = cluster_Kmedoids(grid,n_clusters,data)
    elif algorithm == 'spectral':
        clusters = cluster_Spectral(grid, n_clusters, data)
    elif algorithm == 'hdbscan':
        n_clusters, clusters = cluster_HDBSCAN(grid, n_clusters, data)
    return n_clusters, clusters
    
def _process_clusters(grid, data, cluster_centers, n_clusters, new_columns):
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
        if not hasattr(ts, 'data_clustered') or not isinstance(ts.data_clustered, dict):
            ts.data_clustered = {}
        name = ts.name
        ts.data_clustered[n_clusters] = clusters[name].to_numpy(dtype=float)
    
    return clusters

def cluster_OPTICS(grid, n_clusters, data, min_samples=2, max_eps=np.inf, xi=0.05):
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
    print_clustering_results("OPTICS", actual_clusters, specific_info)
    
    # Process and return results
    processed_results = _process_clusters(grid, data, cluster_centers, actual_clusters, new_columns)
    return actual_clusters, processed_results


def cluster_DBSCAN(grid, n_clusters, data, min_samples=2, initial_eps=0.5):
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
    print_clustering_results("DBSCAN", actual_clusters, specific_info)
    
    # Always call _process_clusters with valid results
    processed_results = _process_clusters(grid, data, cluster_centers, actual_clusters, new_columns)
    return actual_clusters, processed_results

def cluster_Ward(grid, n_clusters, data):
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
    print_clustering_results("Ward hierarchical", n_clusters, specific_info)
    
    processed_results = _process_clusters(grid, data, cluster_centers, n_clusters, new_columns)
    return n_clusters, processed_results

def cluster_Kmeans(grid, n_clusters, data):
    new_columns = data.columns
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    
    # Fit KMeans on scaled data
    kmeans = KMeans(n_clusters=n_clusters)
    data['Cluster'] = kmeans.fit_predict(data_scaled)
    
    # Get cluster centers and transform back to original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Print clustering results
    cluster_sizes = get_cluster_sizes(data)
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Inertia": kmeans.inertia_,
        "Iterations": kmeans.n_iter_
    }
    print_clustering_results("K-means", n_clusters, specific_info)
    
    processed_results = _process_clusters(grid, data, cluster_centers, n_clusters, new_columns)
    return n_clusters, processed_results

def cluster_Kmedoids(grid, n_clusters, data, method='alternate', init='build', max_iter=300):
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
    print_clustering_results("K-medoids", n_clusters, specific_info)
    
    processed_results = _process_clusters(grid, data, cluster_centers, n_clusters, new_columns)
    return n_clusters, processed_results

def cluster_Spectral(grid, n_clusters, data, n_init=10, assign_labels='kmeans', affinity='rbf', gamma=1.0):
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
    print_clustering_results("Spectral", n_clusters, specific_info)
    
    processed_results = _process_clusters(grid, data, cluster_centers, n_clusters, new_columns)
    return n_clusters, processed_results

def cluster_HDBSCAN(grid, n_clusters, data, min_cluster_size=5, min_samples=None, cluster_selection_method='eom'):
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
    
    # Initialize HDBSCAN
    from sklearn.cluster import HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method
    )
    
    # Fit and predict
    data['Cluster'] = clusterer.fit_predict(data_scaled)
    
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
    print_clustering_results("HDBSCAN", actual_clusters, specific_info)
    
    processed_results = _process_clusters(grid, data, cluster_centers, actual_clusters, new_columns)
    return actual_clusters, processed_results

def print_clustering_results(algorithm, n_clusters, specific_info):
    """Helper function to print clustering results in a standardized format."""
    print(f"\n{algorithm} clustering results:")
    print(f"- Number of clusters: {n_clusters}")
    
    # Print algorithm-specific information
    for key, value in specific_info.items():
        if isinstance(value, (int, str)):
            print(f"- {key}: {value}")
        elif isinstance(value, float):
            print(f"- {key}: {value:.2f}")
        elif isinstance(value, list):
            print(f"- {key}: {value}")
            if key == "Cluster sizes":
                print(f"  • Average: {np.mean(value):.1f}")
                print(f"  • Std dev: {np.std(value):.1f}")
        elif isinstance(value, tuple):
            count, percentage = value
            print(f"- {key}: {count} ({percentage:.1f}%)")