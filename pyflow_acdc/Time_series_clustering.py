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
from sklearn.decomposition import PCA



__all__ = ['cluster_TS',
           'cluster_Kmeans',
           'cluster_Ward',
           'cluster_Kmedoids',
           'cluster_PAM_Hierarchical',
           'cluster_DBSCAN',
           'cluster_OPTICS',
           'cluster_Spectral',
           'cluster_HDBSCAN',
           'run_clustering_analysis_and_plot',
           'identify_correlations']


def filter_data(grid, time_series, cv_threshold=0, central_market=[], print_details=False):
    """
    Filter time series data based on type and Coefficient of Variation threshold.
    
    Parameters:
    -----------
    grid : Grid object
        The grid object containing time series
    time_series : list
        List of time series types to include
    cv_threshold : float, default=0
        Minimum Coefficient of Variation threshold. Time series with CV below this 
        will be excluded. CV = std/mean (absolute value)
    central_market : str, default=None
        Central market name. If provided, only time series associated with this market will be included
    print_details : bool, default=True
        If True, print detailed statistics of time series
    Returns:
    --------
    pandas.DataFrame, StandardScaler, pandas.DataFrame
        Filtered scaled data, scaler object, and raw data
    """

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
    if central_market == []:
        central_market = set(grid.Price_Zones_dic.keys())
    PZ_centrals = [grid.Price_Zones[grid.Price_Zones_dic[cm]] for cm in central_market]

    data = pd.DataFrame()
    excluded_ts = []  # Track excluded time series
    columns_to_drop = []
    non_time_series = []
    # First collect all valid time series
    for ts in grid.Time_series:
        name = ts.name
        ts_data = ts.data
        if ts.type in time_series:
            is_in_central = any(ts.TS_num in pz.TS_dict.values() for pz in PZ_centrals)
            if not is_in_central and ts.type in ['price','Load','PGL_min','PGL_max','a_CG','b_CG','c_CG']:
                columns_to_drop.append(name)
        else:
            columns_to_drop.append(name)
            non_time_series.append(name)
        if data.empty:
                data[name] = ts_data
                expected_length = len(ts_data)
        else:
            if len(ts_data) != expected_length:
                print(f"Error: Length mismatch for time series '{name}'. Expected {expected_length}, got {len(ts_data)}. Time series not included")
                continue
            data[name] = ts_data    

    if not data.empty:
        # Calculate and print statistics
        stats = {}
        for column in data.columns:
            mean = np.mean(data[column])
            std = np.std(data[column])
            var = np.var(data[column])
            if mean == 0:
                cv = np.inf if std != 0 else 0
            else:
                cv = abs(std / mean)
            stats[column] = {
                'mean': mean,
                'std': std,
                'var': var,
                'cv': cv
            }
        
        # Print sorted by both CV and variance
        if print_details:
            print("\nTime series statistics (sorted by CV):")
            print(f"{'Name':20} {'Mean':>12} {'Std':>12} {'Var':>12} {'CV':>12}")
            print("-" * 70)
            for column, stat in sorted(stats.items(), key=lambda x: x[1]['cv']):
                print(f"{column:20} {stat['mean']:12.6f} {stat['std']:12.6f} {stat['var']:12.6f} {stat['cv']:12.6f}")
        
        # Filter based on CV threshold
        if cv_threshold > 0:
            for column, stat in stats.items():
                if stat['cv'] < cv_threshold:
                    excluded_ts.append((column, stat['cv']))
                    columns_to_drop.append(column) 

        # Scale the remaining data after filtering
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns
        )

        if columns_to_drop:
            data_scaled = data_scaled.drop(columns=columns_to_drop)
            if print_details:
                print(f"\nExcluded {len(excluded_ts)} time series with CV below {cv_threshold}:")
                for name, cv in excluded_ts:
                    print(f"- {name}: CV = {cv:.6f}")
                print(f"\nExcluded {len(non_time_series)} for being outside of user defined time series: {time_series}")
                for name in non_time_series:
                    print(f"- {name}")
                print(f"\nExcluded {len(columns_to_drop)-len(excluded_ts)-len(non_time_series)} time series not in central market {central_market}:")
                for name in columns_to_drop:
                    if name not in excluded_ts and name not in non_time_series:
                        print(f"- {name}")    
    if print_details:
        if data.empty:
            print("Warning: No time series passed the filtering criteria")
        else:
            print(f"\nIncluded {len(data_scaled.columns)} time series in analysis")

    return data_scaled, scaler, data

def identify_correlations(grid,time_series=[], correlation_threshold=0,cv_threshold=0,central_market=[],print_details=False,corrolation_decisions=[]):
    """
    Identify highly correlated time series variables.
    
    Parameters:
        grid: Grid object containing time series
        correlation_threshold: Correlation coefficient threshold (default: 0.8)
        cv_threshold: Minimum variance threshold (default: 0)
    
    Returns:
        dict: Dictionary containing:
            - correlation_matrix: Full correlation matrix
            - high_correlations: List of tuples (var1, var2, corr_value) for highly correlated pairs
            - groups: List of groups of correlated variables
    """
  
    data_scaled,scaler, data = filter_data(grid,time_series,cv_threshold,central_market,print_details)
    
    if correlation_threshold > 0:
    # Calculate correlation matrix
        corr_matrix = data_scaled.corr()
        
        # Find highly correlated pairs
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > correlation_threshold:
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
        if print_details:
            print(f"\nHighly correlated variables (|correlation| > {correlation_threshold}):")
            for var1, var2, corr in high_corr:
                print(f"{var1:20} - {var2:20}: {corr:.3f}")
        
        if print_details:
            print("\nCorrelated groups:")
            for i, group in enumerate(groups, 1):
                print(f"Group {i}: {', '.join(sorted(group))}")

        #ask user if want to clean correlation groups
        if corrolation_decisions == []:
            clean_groups = input("Do you want to clean correlation groups? (y/n): ")
            if clean_groups == 'y':
                clean_groups = True
                method = input("Choose method (1: highest variance, 2: PCA with new components, 3: PCA representative): ")
                scale_groups = input("Scale by group size to maintain group influence? (y/n): ")
                if scale_groups == 'y':
                    scale_groups = True
                else:
                    scale_groups = False
        else:
            clean_groups = corrolation_decisions[0]
            method = corrolation_decisions[1]
            scale_groups = corrolation_decisions[2]
        columns_to_drop = []

        if clean_groups == True:    
            if method == '1':
                print("\nUsing highest variance method:")
                for group in groups:
                    group_list = list(group)
                    group_variances = data[group_list].var()
                    max_var_col = group_variances.idxmax()
                    print(f"\nGroup: {group_list}")
                    print(f"Variances: {group_variances}")
                    print(f"Keeping: {max_var_col} (variance: {group_variances[max_var_col]:.2f})")
                    
                    if scale_groups:
                        scaling_factor = np.sqrt(len(group_list))
                        print(f"Scaling by sqrt({len(group_list)}) = {scaling_factor:.2f}")
                        data_scaled[max_var_col] *= scaling_factor
                    
                    columns_to_drop.extend([col for col in group_list if col != max_var_col])
            
            elif method == '2':
                print("\nUsing PCA with new components:")
                for group in groups:
                    group_list = list(group)
                    group_data = data_scaled[group_list]
                    
                    print(f"\nGroup: {group_list}")
                    # Apply PCA
                    pca = PCA(n_components=1)
                    pc1 = pca.fit_transform(group_data)
                    
                    # Create new column name
                    new_col = f"PC1_{'_'.join(group_list)}"
                    print(f"Creating new component: {new_col}")
                    print(f"Explained variance ratio: {pca.explained_variance_ratio_[0]:.2%}")
                    
                    # Scale PC if requested
                    if scale_groups:
                        scaling_factor = np.sqrt(len(group_list))
                        print(f"Scaling by sqrt({len(group_list)}) = {scaling_factor:.2f}")
                        pc1 *= scaling_factor
                    
                    data_scaled[new_col] = pc1.ravel()
                    columns_to_drop.extend(group_list)
            
            elif method == '3':
                print("\nUsing PCA representative method:")
                for group in groups:
                    group_list = list(group)
                    group_data = data_scaled[group_list]
                    
                    print(f"\nGroup: {group_list}")
                    # Apply PCA
                    pca = PCA(n_components=1)
                    pc1 = pca.fit_transform(group_data)
                    
                    # Find variable most correlated with PC1
                    correlations = [np.corrcoef(pc1.ravel(), group_data[col])[0,1] for col in group_list]
                    max_cor_idx = np.argmax(np.abs(correlations))
                    max_cor_col = group_list[max_cor_idx]
                    
                    print(f"PC1 explained variance ratio: {pca.explained_variance_ratio_[0]:.2%}")
                    print(f"Correlations with PC1: {dict(zip(group_list, correlations))}")
                    print(f"Keeping: {max_cor_col} (correlation: {correlations[max_cor_idx]:.2f})")
                    
                    if scale_groups:
                        scaling_factor = np.sqrt(len(group_list))
                        print(f"Scaling by sqrt({len(group_list)}) = {scaling_factor:.2f}")
                        data_scaled[max_cor_col] *= scaling_factor
                    
                    columns_to_drop.extend([col for col in group_list if col != max_cor_col])
            
            print(f"\nDropping {len(columns_to_drop)} columns from scaled data: {columns_to_drop}")
            data_scaled = data_scaled.drop(columns=columns_to_drop)
        
    else:
        groups = []
        high_corr = []
        corr_matrix = None
    
    return  [data_scaled,scaler, data], {
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


def cluster_TS(grid, n_clusters, time_series=[],central_market=[], algorithm='Kmeans', cv_threshold=0 ,correlation_threshold=0.8,print_details=False,corrolation_decisions=[]):
    algorithm = algorithm.lower()
    #check if algorithm is valid    
    if algorithm not in {'kmeans','ward','dbscan','optics','kmedoids','spectral','hdbscan'}:
        algorithm='kmeans'
        print(f"Algorithm {algorithm} not found, using Kmeans")
    
        
    [data_scaled,scaler, data],_ = identify_correlations(grid,time_series=time_series, correlation_threshold=correlation_threshold,cv_threshold=cv_threshold,central_market=central_market,print_details=print_details,corrolation_decisions=corrolation_decisions)
  
    if algorithm == 'kmeans':
        clusters, returns, labels = cluster_Kmeans(grid, n_clusters, data, [data_scaled, scaler], print_details=print_details)
    elif algorithm == 'ward':
        clusters, returns, labels = cluster_Ward(grid, n_clusters, data, [data_scaled, scaler], print_details=print_details)
    elif algorithm == 'kmedoids':
        clusters, returns, labels = cluster_Kmedoids(grid, n_clusters, data, [data_scaled, scaler], print_details=print_details)
    elif algorithm == 'pam_hierarchical':
        clusters, returns, labels = cluster_PAM_Hierarchical(grid, n_clusters, data, [data_scaled, scaler], print_details=print_details)
    elif algorithm == 'spectral':
        clusters, returns, labels = cluster_Spectral(grid, n_clusters, data, [data_scaled, scaler], print_details=print_details)
    elif algorithm == 'dbscan':
        n_clusters, clusters, returns, labels = cluster_DBSCAN(grid, n_clusters, data, [data_scaled, scaler], print_details=print_details)
    elif algorithm == 'optics':
        n_clusters, clusters, returns, labels = cluster_OPTICS(grid, n_clusters, data, [data_scaled, scaler], print_details=print_details)    
    elif algorithm == 'hdbscan':
        n_clusters, clusters, returns, labels = cluster_HDBSCAN(grid, n_clusters, data, [data_scaled, scaler], print_details=print_details)
    

    return n_clusters, clusters, returns, labels

def _process_clusters(grid, data, cluster_centers):
    """
    Process clustering results and update grid with cluster information.
    
    Parameters:
    -----------
    grid : pandapower.Grid
        The power system grid object to be updated with cluster information
    data : pandas.DataFrame
        Time series data used for clustering
    cluster_centers : numpy.ndarray
        Array containing the centroids/medoids of each cluster
        
    Returns:
    --------
    grid : pandapower.Grid
        Updated grid object with cluster information
    """
    new_columns = [col for col in data.columns if col != 'Cluster']
    n_clusters = len(cluster_centers)
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


def cluster_Kmedoids(grid, n_clusters, data, scaling_data =None, method='alternate', init='build', max_iter=300,print_details=False):
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
    if scaling_data is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        [data_scaled, scaler] = scaling_data
    
    # Fit KMedoids on scaled data
    kmedoids = KMedoids(
        n_clusters=n_clusters,
        method=method,
        init=init,
        max_iter=max_iter
    )
    labels = kmedoids.fit_predict(data_scaled)
    # Get medoid indices
    medoid_indices = kmedoids.medoid_indices_
    # Get cluster centers (medoids) in original scale
    cluster_centers = data.iloc[medoid_indices].values  
    
    # Print clustering results
    cluster_sizes = pd.Series(labels).value_counts().sort_index().values
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Method": method,
        "Initialization": init,
        "Inertia": kmedoids.inertia_
    }
    if print_details:
        CoV = print_clustering_results("K-medoids", n_clusters, specific_info)
    else:
        # Calculate CoV from cluster sizes
        cluster_sizes = specific_info["Cluster sizes"]
        CoV = np.std(cluster_sizes)/np.mean(cluster_sizes)
    
    data['Cluster'] = labels
    processed_results = _process_clusters(grid, data, cluster_centers)
    return  processed_results, [CoV,kmedoids.inertia_], [data_scaled,labels]

def cluster_Kmeans(grid, n_clusters, data, scaling_data=None, print_details=False):
    """
    Perform K-means clustering on the data.
    
    Parameters:
    -----------
    grid : Grid object
        The grid object to update
    n_clusters : int
        Number of clusters
    data : pandas.DataFrame
        Data to cluster (filtered columns only)
    scaling_data : tuple, optional
        (scaled_data, scaler) if already scaled
    ts_types : list
        List of time series types
    print_details : bool
        Whether to print clustering details
    """
    if scaling_data is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        [data_scaled, scaler] = scaling_data
    
    # Fit KMeans on scaled data (filtered columns)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data_scaled)
    
    all_centers = []
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_means = data[cluster_mask].mean()
        all_centers.append(cluster_means)
    
    cluster_centers = np.array(all_centers)
    
    # Print clustering results
    if print_details:
        cluster_sizes = pd.Series(labels).value_counts().sort_index().values
        specific_info = {
            "Cluster sizes": cluster_sizes,
            "Inertia": kmeans.inertia_,
            "n_iter": kmeans.n_iter_
        }
        CoV = print_clustering_results("K-means", n_clusters, specific_info)
    else:
        cluster_sizes = pd.Series(labels).value_counts().values
        CoV = np.std(cluster_sizes)/np.mean(cluster_sizes)

    data['Cluster'] = labels
    processed_results = _process_clusters(grid, data, cluster_centers)
    return processed_results, [CoV, kmeans.inertia_, kmeans.n_iter_], [data_scaled, labels]

def cluster_Ward(grid, n_clusters, data, scaling_data=None, print_details=False):
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
    if scaling_data is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        [data_scaled, scaler] = scaling_data
    
    # Fit clustering
    ward = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward',
        compute_distances=True
    )
    labels = ward.fit_predict(data_scaled)
    
    # Calculate cluster centers
    all_centers = []
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_means = data[cluster_mask].mean()
        all_centers.append(cluster_means)
    
    cluster_centers = np.array(all_centers)
    
    # Get cluster sizes
    cluster_sizes = pd.Series(labels).value_counts().sort_index().values
    
    # Get additional metrics
    distances = ward.distances_
    
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Maximum merge distance": float(max(distances)) if len(distances) > 0 else 0,
        "Average merge distance": float(np.mean(distances)) if len(distances) > 0 else 0
    }
    if print_details:
        CoV = print_clustering_results("Ward hierarchical", n_clusters, specific_info)
    else:
        CoV = np.std(cluster_sizes)/np.mean(cluster_sizes)
    
    data['Cluster'] = labels
    processed_results = _process_clusters(grid, data, cluster_centers)
    return processed_results, CoV, [data_scaled, labels]

def cluster_PAM_Hierarchical(grid, n_clusters, data, scaling_data=None, print_details=False):
    """
    Perform PAM-based hierarchical clustering using AgglomerativeClustering.
    
    Parameters:
    -----------
    grid : Grid object
        The grid object to update
    n_clusters : int
        Number of clusters
    data : pandas.DataFrame
        Data to cluster
    scaling_data : tuple, optional
        (scaled_data, scaler) if already scaled
    print_details : bool
        Whether to print clustering details
    """
    if scaling_data is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        [data_scaled, scaler] = scaling_data
    
    # Fit clustering using manhattan distance (typical for PAM)
    HierarchicalMedoid = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average',
        metric='manhattan',
        compute_distances=True
    )
    labels = HierarchicalMedoid.fit_predict(data_scaled)
    
    # Find medoid indices for all clusters
    medoid_indices = []
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_data = data[cluster_mask]
        if len(cluster_data) > 0:
            distances = pairwise_distances(
                cluster_data, 
                metric='manhattan'
            )
            medoid_idx = cluster_data.index[distances.sum(axis=1).argmin()]
            medoid_indices.append(medoid_idx)
    
    # Get cluster centers using medoid indices
    cluster_centers = data.iloc[medoid_indices].values
    
    # Get cluster sizes
    cluster_sizes = pd.Series(labels).value_counts().sort_index().values
    
    # Get additional metrics
    distances = HierarchicalMedoid.distances_
    
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Maximum merge distance": float(max(distances)) if len(distances) > 0 else 0,
        "Average merge distance": float(np.mean(distances)) if len(distances) > 0 else 0
    }
    if print_details:
        CoV = print_clustering_results("PAM hierarchical", n_clusters, specific_info)
    else:
        CoV = np.std(cluster_sizes)/np.mean(cluster_sizes)
    
    data['Cluster'] = labels
    processed_results = _process_clusters(grid, data, cluster_centers)
    return processed_results, CoV, [data_scaled, labels]

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

def run_clustering_analysis(grid, save_path='clustering_results',algorithms = ['kmeans', 'kmedoids', 'ward', 'dbscan', 'hdbscan'],n_clusters_list = [1, 4, 8, 16, 24, 48],time_series=[],print_details=False,corrolation_decisions=[True,'2',True]):
       
    
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
                _,_,CoV,info = cluster_TS(grid, algorithm=algo, n_clusters=n,time_series=time_series,print_details=print_details,corrolation_decisions=corrolation_decisions)
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
                            format='svg',print_details=False):
    """
    Plot two time series with their cluster assignments in different colors.
    """
    # Get clusters
    n_clusters, clusters, returns, labels = cluster_TS(
        grid, number_of_clusters,time_series=take_into_account_time_series, algorithm=algorithm,print_details=False)
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
    plt.show()
    plt.close()



def cluster_OPTICS(grid, n_clusters, data, scaling_data=None, min_samples=2, max_eps=np.inf, xi=0.05, print_details=False): 
 
    if scaling_data is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        [data_scaled, scaler] = scaling_data
    
    # Try different xi values until we get desired number of clusters
    best_labels = None
    best_xi = None
    current_xi = xi
    
    while current_xi <= 1.0:
        optics = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=current_xi)
        labels = optics.fit_predict(data_scaled)
        
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
    
    # Calculate cluster centers (medoids) from original data
    all_centers = []
    for i in range(actual_clusters):
        cluster_mask = best_labels == i
        cluster_data = data[cluster_mask]
        medoid_idx = find_medoid(cluster_data)
        all_centers.append(data.loc[medoid_idx])
    
    cluster_centers = np.array(all_centers)
    
    # Get cluster sizes and noise info
    cluster_sizes = pd.Series(best_labels).value_counts().sort_index().values
    noise_points = len(data[best_labels == -1])
    noise_percentage = (noise_points / len(data)) * 100
    
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Found clusters": actual_clusters,
        "Maximum allowed": n_clusters,
        "Final xi": best_xi,
        "Noise points": (noise_points, noise_percentage)
    }
    if print_details:
        CoV = print_clustering_results("OPTICS", actual_clusters, specific_info)
    else:
        CoV = np.std(cluster_sizes)/np.mean(cluster_sizes)
    
    data['Cluster'] = best_labels
    processed_results = _process_clusters(grid, data, cluster_centers)
    return actual_clusters, processed_results, CoV, [data_scaled, best_labels]

def cluster_DBSCAN(grid, n_clusters, data, scaling_data=None, min_samples=2, initial_eps=0.5, print_details=False):
    """
    [Previous docstring remains the same]
    """
    if scaling_data is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        [data_scaled, scaler] = scaling_data
    
    eps = initial_eps
    best_labels = None
    best_eps = None
    
    while eps <= 10.0:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_scaled)
        
        actual_clusters = len(set(labels[labels >= 0]))
        
        if actual_clusters > 0:
            if actual_clusters <= n_clusters:
                best_labels = labels
                best_eps = eps
                break
            else:
                eps *= 1.1
        else:
            eps *= 1.5
    
    if best_labels is None:
        print("Warning: Could not find any meaningful clusters. Try adjusting parameters.")
        return 0, None
    
    # Calculate cluster centers (medoids) from original data
    all_centers = []
    for i in range(len(set(best_labels[best_labels >= 0]))):
        cluster_mask = best_labels == i
        cluster_data = data[cluster_mask]
        medoid_idx = find_medoid(cluster_data)
        all_centers.append(data.loc[medoid_idx])
    
    cluster_centers = np.array(all_centers)
    
    # Get cluster sizes and noise info
    cluster_sizes = pd.Series(best_labels).value_counts().sort_index().values
    noise_points = len(data[best_labels == -1])
    noise_percentage = (noise_points / len(data)) * 100
    
    specific_info = {
        "Cluster sizes": cluster_sizes,
        "Found clusters": actual_clusters,
        "Maximum allowed": n_clusters,
        "Final eps": best_eps,
        "Noise points": (noise_points, noise_percentage)
    }
    if print_details:
        CoV = print_clustering_results("DBSCAN", actual_clusters, specific_info)
    else:
        CoV = np.std(cluster_sizes)/np.mean(cluster_sizes)
    
    data['Cluster'] = best_labels
    processed_results = _process_clusters(grid, data, cluster_centers)
    return actual_clusters, processed_results, CoV, [data_scaled, best_labels]


def cluster_Spectral(grid, n_clusters, data, scaling_data=None, n_init=10, assign_labels='kmeans', affinity='rbf', gamma=1.0, print_details=False):
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
    if scaling_data is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        [data_scaled, scaler] = scaling_data
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        n_init=n_init,
        assign_labels=assign_labels,
        affinity=affinity,
        gamma=gamma,
        random_state=42
    )
    
    labels = spectral.fit_predict(data_scaled)
    
    # Calculate cluster centers (medoids) from original data
    all_centers = []
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_data = data[cluster_mask]
        medoid_idx = find_medoid(cluster_data)
        all_centers.append(data.loc[medoid_idx])
    
    cluster_centers = np.array(all_centers)
    
    # Get cluster sizes and affinity info
    cluster_sizes = pd.Series(labels).value_counts().sort_index().values
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
    if print_details:
        CoV = print_clustering_results("Spectral", n_clusters, specific_info)
    else:
        CoV = np.std(cluster_sizes)/np.mean(cluster_sizes)
    
    data['Cluster'] = labels
    processed_results = _process_clusters(grid, data, cluster_centers)
    return processed_results, CoV, [data_scaled, labels]

def cluster_HDBSCAN(grid, n_clusters, data, scaling_data=None, min_cluster_size=5, min_samples=None, cluster_selection_method='eom', print_details=False):
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
    if scaling_data is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        [data_scaled, scaler] = scaling_data

  
    
    # If min_samples not specified, use min_cluster_size
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Initialize HDBSCA
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method
    )
    
    labels = clusterer.fit_predict(data_scaled)
    actual_clusters = len(set(labels[labels >= 0]))
    
    # Calculate cluster centers (medoids) from original data
    all_centers = []
    for i in range(actual_clusters):
        cluster_mask = labels == i
        cluster_data = data[cluster_mask]
        medoid_idx = find_medoid(cluster_data)
        all_centers.append(data.loc[medoid_idx])
    
    cluster_centers = np.array(all_centers)
    
    # Get cluster sizes and noise info
    cluster_sizes = pd.Series(labels).value_counts().sort_index().values
    noise_points = len(data[labels == -1])
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
    if print_details:
        CoV = print_clustering_results("HDBSCAN", actual_clusters, specific_info)
    else:
        CoV = np.std(cluster_sizes)/np.mean(cluster_sizes)
    
    data['Cluster'] = labels
    processed_results = _process_clusters(grid, data, cluster_centers)
    return actual_clusters, processed_results, CoV, [data_scaled, labels]

def find_medoid(cluster_data):
    """Helper function to find medoid of a cluster"""
    distances = pairwise_distances(cluster_data, metric='manhattan')
    medoid_idx = distances.sum(axis=1).argmin()
    return cluster_data.index[medoid_idx]




