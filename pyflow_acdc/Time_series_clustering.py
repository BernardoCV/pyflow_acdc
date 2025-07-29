from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering, HDBSCAN
from sklearn.metrics import pairwise_distances,davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from kmedoids import KMedoids
import os
if 'MPLBACKEND' not in os.environ:
    os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as time
from pathlib import Path
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

# Plotting constants
FIGURE_WIDTH_CM = 8.25
FIGURE_RATIO = 6/10
CM_TO_INCHES = 2.54
LARGE_DATA_THRESHOLD = 10000
SCATTER_SIZE = 16
SAVE_DPI = 300
CORRELATION_FIGSIZE = (12, 10)
LABEL_ROTATION = 90
LEGEND_Y_POSITION = 1.15

# Algorithm constants
MAX_ITERATIONS = 300
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 10
DEFAULT_MIN_SAMPLES = 2
DEFAULT_XI = 0.05
DEFAULT_GAMMA = 1.0

# Default parameters
DEFAULT_CLUSTER_NUMBERS = [1, 4, 8, 16, 24, 48]
DEFAULT_TS_OPTIONS = [None, 0, 0.8]
DEFAULT_CORRELATION_DECISIONS = [True, '2', True]

# Additional algorithm constants
DEFAULT_CORRELATION_THRESHOLD = 0.8
DEFAULT_CV_THRESHOLD = 0
DEFAULT_SEASONAL_PERIOD_HOURS = 168
DEFAULT_TIME_RESOLUTION_HOURS = 1

# DBSCAN/OPTICS constants
DBSCAN_MAX_EPS = 10.0
DBSCAN_EPS_MULTIPLIER = 1.1
DBSCAN_EPS_INCREASE = 1.5
OPTICS_XI_INCREASE = 1.5
OPTICS_XI_DECREASE = 0.8

# Plotting constants
TICK_LENGTH = 3
TICK_WIDTH = 0.8
INERTIA_PLOT_LIMIT = 100000

def _prepare_scaled_data(data, scaling_data):
    if scaling_data is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        [data_scaled, scaler] = scaling_data
    return data_scaled, scaler

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
                #'c_CG',     # Price zone cost generation parameter c
                'PGL_min',  # Price zone minimum generation limit
                'PGL_max',  # Price zone maximum generation limit
                'price',    # Price for price zones and AC nodes
                'Load',     # Load factor for price zones and AC nodes
                'WPP',      # Wind Power Plant availability
                'OWPP',     # Offshore Wind Power Plant availability
                'SF',       # Solar Farm availability
                'REN'       # Generic Renewable source availability
            ]
    if central_market == [] or central_market is  None:
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

def  identify_correlations(grid,time_series=[], correlation_threshold=0,cv_threshold=0,central_market=[],print_details=False,correlation_decisions=[]):
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
        
        
        high_corr = []
        
        corr_stack = corr_matrix.stack()
        
        upper_triangle = corr_stack[corr_stack.index.get_level_values(0) < corr_stack.index.get_level_values(1)]
        
        high_corr_filtered = upper_triangle[abs(upper_triangle) > correlation_threshold]

        
        high_corr = [(var1, var2, abs(corr)) for (var1, var2), corr in high_corr_filtered.items()]
        
        
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
        if correlation_decisions == []:
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
            clean_groups = correlation_decisions[0]
            method = correlation_decisions[1]
            scale_groups = correlation_decisions[2]
        columns_to_drop = []

        if clean_groups:    
            if method == '1' or method == 1:
                
                for group in groups:
                    group_list = list(group)
                    group_variances = data[group_list].var()
                    max_var_col = group_variances.idxmax()
                    if print_details:
                        print("\nUsing highest variance method:")
                        print(f"\nGroup: {group_list}")
                        print(f"Variances: {group_variances}")
                        print(f"Keeping: {max_var_col} (variance: {group_variances[max_var_col]:.2f})")
                    
                    if scale_groups:
                        scaling_factor = np.sqrt(len(group_list))
                        print(f"Scaling by sqrt({len(group_list)}) = {scaling_factor:.2f}")
                        data_scaled[max_var_col] *= scaling_factor
                    
                    columns_to_drop.extend([col for col in group_list if col != max_var_col])
            
            elif method == '2' or method == 2:
                
                for group in groups:
                    group_list = list(group)
                    group_data = data_scaled[group_list]
                    
                    # Apply PCA
                    pca = PCA(n_components=1)
                    pc1 = pca.fit_transform(group_data)
                    
                    # Create new column name
                    new_col = f"PC1_{'_'.join(group_list)}"
                    if print_details:
                        print("\nUsing PCA with new components:")
                        print(f"\nGroup: {group_list}")
                        print(f"Creating new component: {new_col}")
                        print(f"Explained variance ratio: {pca.explained_variance_ratio_[0]:.2%}")
                    
                    # Scale PC if requested
                    if scale_groups:
                        scaling_factor = np.sqrt(len(group_list))
                        if print_details:
                            print(f"Scaling by sqrt({len(group_list)}) = {scaling_factor:.2f}")
                        pc1 *= scaling_factor
                    
                    data_scaled[new_col] = pc1.ravel()
                    columns_to_drop.extend(group_list)
            
            elif method == '3' or method == 3:
                
                for group in groups:
                    group_list = list(group)
                    group_data = data_scaled[group_list]
                    
                    
                    # Apply PCA
                    pca = PCA(n_components=1)
                    pc1 = pca.fit_transform(group_data)
                    
                    # Find variable most correlated with PC1
                    correlations = [np.corrcoef(pc1.ravel(), group_data[col])[0,1] for col in group_list]
                    max_cor_idx = np.argmax(np.abs(correlations))
                    max_cor_col = group_list[max_cor_idx]
                    
                    if print_details:
                        print("\nUsing PCA representative method:")
                        print(f"\nGroup: {group_list}")
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

def plot_time_series(data,labels,var_name,n_clusters,save_path,format,identifier,algo):
    # Convert 8.25 cm to inches and maintain ratio
    width_inches = FIGURE_WIDTH_CM / CM_TO_INCHES
    if len(data) > LARGE_DATA_THRESHOLD:
        width_inches = width_inches * 2
    # Set publication-quality plotting parameters
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (width_inches, width_inches*FIGURE_RATIO),
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'lines.markersize': 4,
        'lines.linewidth': 1,
        'grid.alpha': 0.3
    })

    
    # Create figure and apply formatting
    fig, ax = plt.subplots()
    max_colors = 8  # Set2 colormap has 8 distinct colors
    colors = plt.cm.Set2(np.linspace(0, 1, max_colors))
    markers = ['+', 'x', '*', '^', 'v', '<', '>', 's']  # Backup markers when colors repeat
    
    # Plot clusters with consistent styling
    for i in range(n_clusters):
        mask = labels == i
        time_points = np.arange(len(data))[mask]
        values = data.loc[mask, var_name]
        
        # If we've exceeded the number of colors, start cycling markers
        current_marker = '+' if i < max_colors else markers[((i - max_colors) % len(markers))]
        
        ax.scatter(time_points, values, 
                    marker=current_marker,
                    color=colors[i % max_colors],
                    alpha=.8, 
                    s=SCATTER_SIZE)
    
    # Format axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(direction='out', length=TICK_LENGTH, width=TICK_WIDTH)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Value (standardized) of {var_name}')
    ax.set_title(f'Time Series Clustering\n{algo}, {n_clusters} clusters')
    
    plt.tight_layout()
    
    # Save plot with consistent settings
    plt.savefig(f'{save_path}/timeseries_clustering_{algo}_{n_clusters}_{identifier}.{format}', 
                dpi=SAVE_DPI,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()


def plot_correlation_matrix(corr_matrix, save_path=None):
    """
    Plot correlation matrix as a heatmap.
    
    Parameters:
        corr_matrix: Pandas DataFrame with correlation matrix
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=CORRELATION_FIGSIZE)
    
    # Create heatmap
    plt.imshow(corr_matrix, cmap='RdBu', aspect='equal', vmin=-1, vmax=1)
    
    # Add labels
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=LABEL_ROTATION)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    
    # Vectorized text placement
    n_cols = len(corr_matrix.columns)
    for i, j in np.ndindex(n_cols, n_cols):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                ha='center', va='center')
    
    plt.title('Correlation Matrix of Time Series')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def cluster_TS(grid, n_clusters, time_series=[], central_market=[], algorithm='Kmeans', 
              cv_threshold=0, correlation_threshold=0.8, print_details=False, 
              correlation_decisions=[], **kwargs):
    """
    Main clustering function with enhanced parameter support.
    
    Additional Parameters:
    --------------------
    **kwargs : dict
        Additional parameters passed to specific clustering algorithms:
        - For kmedoids: method, init, max_iter, random_state, metric
        - For kmeans: random_state, n_init, max_iter
        - For spectral: affinity, gamma, assign_labels
        - For dbscan: eps, min_samples
        - For optics: min_samples, max_eps, xi
        - For hdbscan: min_cluster_size, min_samples, cluster_selection_method
    """
    algorithm = algorithm.lower()
    valid_algorithms = {'kmeans','ward','dbscan','optics','spectral','hdbscan','pam_hierarchical','kmedoids'}

    if algorithm not in valid_algorithms:
        print(f"Algorithm {algorithm} not found, using Kmeans")
        algorithm='kmeans'
    
    [data_scaled,scaler, data],_ = identify_correlations(grid,time_series=time_series, 
                                                        correlation_threshold=correlation_threshold,
                                                        cv_threshold=cv_threshold,
                                                        central_market=central_market,
                                                        print_details=print_details,
                                                        correlation_decisions=correlation_decisions)
  
    if algorithm == 'kmeans':
        clusters, returns, data_info = cluster_Kmeans(grid, n_clusters, data, [data_scaled, scaler], 
                                                     print_details=print_details, **kwargs)
    elif algorithm == 'ward':
        clusters, returns, data_info = cluster_Ward(grid, n_clusters, data, [data_scaled, scaler], 
                                                   print_details=print_details, **kwargs)
    elif algorithm == 'kmedoids':
        clusters, returns, data_info = cluster_Kmedoids(grid, n_clusters, data, [data_scaled, scaler], 
                                                       print_details=print_details, **kwargs)
    elif algorithm == 'pam_hierarchical':
        clusters, returns, data_info = cluster_PAM_Hierarchical(grid, n_clusters, data, [data_scaled, scaler], 
                                                               print_details=print_details, **kwargs)
    elif algorithm == 'spectral':
        clusters, returns, data_info = cluster_Spectral(grid, n_clusters, data, [data_scaled, scaler], 
                                                       print_details=print_details, **kwargs)
    elif algorithm == 'dbscan':
        n_clusters, clusters, returns, data_info = cluster_DBSCAN(grid, n_clusters, data, [data_scaled, scaler], 
                                                                 print_details=print_details, **kwargs)
    elif algorithm == 'optics':
        n_clusters, clusters, returns, data_info = cluster_OPTICS(grid, n_clusters, data, [data_scaled, scaler], 
                                                                 print_details=print_details, **kwargs)    
    elif algorithm == 'hdbscan':
        n_clusters, clusters, returns, data_info = cluster_HDBSCAN(grid, n_clusters, data, [data_scaled, scaler], 
                                                                  print_details=print_details, **kwargs)
    
    data_scaled, labels = data_info
    data_info = [data, data_scaled, labels]
    return n_clusters, clusters, returns, data_info

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


def cluster_Kmedoids(grid, n_clusters, data, scaling_data =None, method='alternate', 
                    init='build', max_iter=MAX_ITERATIONS, print_details=False, 
                    random_state=None, metric='euclidean'):
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
    scaling_data : tuple, optional
        (scaled_data, scaler) if already scaled
    method : str, default='alternate'
        {'alternate', 'pam', 'fasterpam', 'fastpam1'} Algorithm to use
    init : str, default='build'
        {'random', 'heuristic', 'k-medoids++', 'build', 'first'} Initialization method
    max_iter : int, default=MAX_ITERATIONS
        Maximum number of iterations
    print_details : bool, default=False
        Whether to print detailed clustering information
    random_state : int, optional
        Random state for reproducibility
    metric : str, default='euclidean'
        Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
    """
    data_scaled, scaler = _prepare_scaled_data(data, scaling_data)
    
    # Fit KMedoids on scaled data
    kmedoids = KMedoids(
        n_clusters=n_clusters,
        method=method,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        metric=metric
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
        "Metric": metric,
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
    return processed_results, [CoV, kmedoids.inertia_], [data_scaled, labels]


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
    data_scaled, scaler = _prepare_scaled_data(data, scaling_data)
    
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
    data_scaled, scaler = _prepare_scaled_data(data, scaling_data)
    
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
    data_scaled, scaler = _prepare_scaled_data(data, scaling_data)
    
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

def run_clustering_analysis(grid, save_path='clustering_results',algorithms = ['kmeans', 'kmedoids', 'ward', 'pam_hierarchical'],n_clusters_list = DEFAULT_CLUSTER_NUMBERS,time_series=[],print_details=False,ts_options=[None,0,0.8],correlation_decisions=[True,'2',True],plotting=False, plotting_options=[None,'.png'],identifier=None):
    
    results = {
        'algorithm': [],
        'n_clusters': [],
        'time_taken': [],
        'Coefficient of Variation': [],
        'inertia': [],
        'davies_bouldin_combined': [],
        'davies_bouldin_value': [],
        'davies_bouldin_seasonal': []
    }
    
    for algo in algorithms:
        print(f"\nTesting {algo}...")
        for n in n_clusters_list:
            print(f"  Clusters: {n}")
            
            start_time = time.time()
            try:
                
                n_clusters,_,CoV,data_info = cluster_TS(grid, n_clusters= n, time_series=time_series,central_market=ts_options[0],algorithm=algo, cv_threshold=ts_options[1] ,correlation_threshold=ts_options[2],print_details=print_details,correlation_decisions=correlation_decisions)
                data,data_scaled,labels = data_info
                if algo == 'kmeans':
                    CoV, inertia, n_iter_ = CoV
                elif algo == 'kmedoids':
                    CoV, inertia = CoV
                else:
                    inertia = 0
                time_taken = time.time() - start_time

                
                metrics = evaluate_clustering(data_scaled, labels)
                db_score        = metrics['davies_bouldin_combined']
                value_db_score  = metrics['davies_bouldin_value']
                season_db_score = metrics['davies_bouldin_seasonal']

                if plotting:
                    if plotting_options[0] is None:
                        covs = np.std(data_scaled, axis=0) / np.mean(np.abs(data_scaled), axis=0)
                        highest_cov_idx = np.argmax(covs)
                        var_name = data.columns[highest_cov_idx]

                        print(f"Plotting time series with highest CoV: {var_name} (CoV = {covs.iloc[highest_cov_idx]:.3f})")
                    else:
                        var_name = plotting_options[0]
                        highest_cov_idx = data.columns.get_loc(var_name)
                        print(f"Plotting specified time series: {var_name}")

                    plot_time_series(data,labels,var_name,n_clusters,save_path,plotting_options[1],identifier,algo)    
            
                results['algorithm'].append(algo)
                results['n_clusters'].append(n)
                results['time_taken'].append(time_taken)
                results['Coefficient of Variation'].append(CoV)
                results['inertia'].append(inertia)
                results['davies_bouldin_combined'].append(db_score)
                results['davies_bouldin_value'].append(value_db_score)
                results['davies_bouldin_seasonal'].append(season_db_score)
                
                print(f"    Time: {time_taken:.2f}s")
                
            except Exception as e:
                print(f"    Error with {algo}, n={n}: {str(e)}")
                continue
    
    df_results = pd.DataFrame(results)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Updated summary to use correct columns
    summary_df = df_results[['algorithm', 'n_clusters', 'time_taken', 'Coefficient of Variation','inertia','davies_bouldin_combined','davies_bouldin_value','davies_bouldin_seasonal']]
    summary_df.to_csv(f'{save_path}/clustering_summary_{identifier}.csv', index=False)
 
    
    return df_results

# Usage:
# results = run_clustering_analysis(grid)

# To analyze results:
def plot_clustering_results(df=None, results_path='clustering_results', format='svg',identifier=None):
    """
    Plot clustering analysis results with publication-quality formatting.
    
    Parameters:
    -----------
    df : pandas.DataFrame, optional
        DataFrame containing clustering results. If None, loads from results_path
    results_path : str, default='clustering_results'
        Path to save the generated plots
    format : str, default='svg'
        Output format for plots ('svg', 'png', etc.)
    """
    # Convert 8.25 cm to inches and maintain ratio
    width_inches = FIGURE_WIDTH_CM / CM_TO_INCHES
    ratio = 6/10  # Original height/width ratio
    height_inches = width_inches * ratio
    
    # Set publication-quality plotting parameters
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (width_inches, height_inches),
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'lines.markersize': 4,
        'lines.linewidth': 1,
        'grid.alpha': 0.3
    })
    
    if df is None:
        df = pd.read_csv(f'{results_path}/clustering_summary.csv')
    
    def format_axes(ax):
        """Apply consistent formatting to plot axes"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(direction='out', length=TICK_LENGTH, width=TICK_WIDTH)
    
    # Define consistent color palette
    algorithms = df['algorithm'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # Create plots with consistent styling
    metrics = [
        ('time_taken', 'Time (seconds)',10),
        ('Coefficient of Variation', 'Coefficient of Variation',2),
        ('inertia', 'Inertia',INERTIA_PLOT_LIMIT),
        ('davies_bouldin_combined', 'Davies-Bouldin Index',2.5),
        ('davies_bouldin_value',    'Davies-Bouldin Index',2.5),
        ('davies_bouldin_seasonal', 'Davies-Bouldin Index',2.5)
    ]
    
    for metric, ylabel,ymax in metrics:
        fig, ax = plt.subplots()
        
        for idx, algo in enumerate(algorithms):
            data = df[df['algorithm'] == algo]
            ax.plot(data['n_clusters'], data[metric],
                   marker=markers[idx % len(markers)],
                   color=colors[idx],
                   label=algo,
                   markersize=4,
                   linewidth=1)
        
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel(ylabel)
        ax.set_ylim(0,ymax)

        # Adjust legend
        ax.legend(bbox_to_anchor=(0.5, LEGEND_Y_POSITION),
                 loc='upper center',
                 ncol=2,
                 frameon=False)
        
        format_axes(ax)
        plt.tight_layout()
        
        # Save plot
        metric_name = metric.lower().replace('_', '-')
        plt.savefig(f'{results_path}/{metric_name}-comparison_{identifier}.{format}',
                   dpi=SAVE_DPI,
                   bbox_inches='tight',
                   pad_inches=0.1)
        plt.close()

def run_clustering_analysis_and_plot(grid,algorithms = ['kmeans', 'kmedoids', 'ward', 'pam_hierarchical'],n_clusters_list = DEFAULT_CLUSTER_NUMBERS,path='clustering_results',time_series=[],print_details=False,ts_options=[None,0,0.8],correlation_decisions=[True,'2',True],plotting_options=[None,'svg'],identifier=None):
    
    results = run_clustering_analysis(grid,path,algorithms,n_clusters_list,time_series,print_details,ts_options,correlation_decisions,plotting=True, plotting_options=plotting_options,identifier=identifier)
    plot_clustering_results(results,path,format=plotting_options[1],identifier=identifier)

def Time_series_cluster_relationship(grid, ts1_name=None, ts2_name=None,price_zone=None,ts_type=None, algorithm='kmeans', 
                            take_into_account_time_series=[], 
                            number_of_clusters=2, path='clustering_results', 
                            format='svg',print_details=False):
    """
    Plot two time series with their cluster assignments in different colors.
    """
    # Get clusters
    n_clusters, clusters, returns, data_info = cluster_TS(
        grid, number_of_clusters,time_series=take_into_account_time_series, algorithm=algorithm,print_details=False)
    data,data_scaled,labels = data_info

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
    width_inches = FIGURE_WIDTH_CM / CM_TO_INCHES
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



def cluster_OPTICS(grid, n_clusters, data, scaling_data=None, min_samples=DEFAULT_MIN_SAMPLES, 
                  max_eps=np.inf, xi=DEFAULT_XI, print_details=False):
 
    data_scaled, scaler = _prepare_scaled_data(data, scaling_data)
    
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
            current_xi *= OPTICS_XI_INCREASE  # Increase xi to get fewer clusters
        else:  # No clusters found
            current_xi *= OPTICS_XI_DECREASE  # Decrease xi to get more clusters
    
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

def cluster_DBSCAN(grid, n_clusters, data, scaling_data=None, min_samples=DEFAULT_MIN_SAMPLES, initial_eps=0.5, print_details=False):
    """
    [Previous docstring remains the same]
    """
    data_scaled, scaler = _prepare_scaled_data(data, scaling_data)
    
    eps = initial_eps
    best_labels = None
    best_eps = None
    
    while eps <= DBSCAN_MAX_EPS:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_scaled)
        
        actual_clusters = len(set(labels[labels >= 0]))
        
        if actual_clusters > 0:
            if actual_clusters <= n_clusters:
                best_labels = labels
                best_eps = eps
                break
            else:
                eps *= DBSCAN_EPS_MULTIPLIER
        else:
            eps *= DBSCAN_EPS_INCREASE
    
    if best_labels is None:
        print("Warning: Could not find any meaningful clusters. Try adjusting parameters.")
        return 0, None
    
    # Calculate cluster centers (medoids) from original data
    all_centers = []
    valid_labels = best_labels[best_labels >= 0]
    unique_clusters = np.unique(valid_labels)
    for i in range(len(unique_clusters)):
        cluster_id = unique_clusters[i]
        cluster_mask = best_labels == cluster_id
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
    n_init : int, default=N_INIT
        Number of times the k-means algorithm will be run with different centroid seeds
    assign_labels : {'kmeans', 'discretize'}, default='kmeans'
        Strategy to assign labels in the embedding space
    affinity : {'rbf', 'nearest_neighbors', 'precomputed'}, default='rbf'
        How to construct the affinity matrix
    gamma : float, default=1.0
        Kernel coefficient for rbf kernel
    """
    data_scaled, scaler = _prepare_scaled_data(data, scaling_data)
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        n_init=n_init,
        assign_labels=assign_labels,
        affinity=affinity,
        gamma=gamma,
        random_state=DEFAULT_RANDOM_STATE
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
    data_scaled, scaler = _prepare_scaled_data(data, scaling_data)

  
    
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



def evaluate_clustering(data_scaled, labels, time_resolution_hours=DEFAULT_TIME_RESOLUTION_HOURS, 
                        seasonal_period_hours=DEFAULT_SEASONAL_PERIOD_HOURS):
    """
    Evaluate time series clustering using standard DB index plus temporal and seasonal components.
    
    Parameters:
    -----------
    data_scaled : array-like
        Scaled data for clustering
    labels : array-like
        Cluster labels
    time_resolution_hours : float, default=1
        Time resolution of the data in hours (e.g., 0.25 for 15-min, 1 for hourly, 24 for daily)
    seasonal_period_hours : float, default=168
        Seasonal period to analyze in hours (e.g., 168 for weekly, 8760 for yearly)
    """
    def temporal_seasonal_scores(X, labels):
        """Calculate seasonal component using DB-style calculation"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Calculate cluster centers for seasonal patterns
        seasonal_centers = []
        
        for label in unique_labels:
            cluster_points = X[labels == label]
            
            # Seasonal pattern - now configurable
            seasonal_period_points = int(seasonal_period_hours / time_resolution_hours)
            if cluster_points.shape[0] > seasonal_period_points:
                # Reshape to seasonal periods
                periods = cluster_points.shape[0] // seasonal_period_points * seasonal_period_points
                reshaped = cluster_points[:periods].reshape(-1, seasonal_period_points, X.shape[1])
                seasonal_pattern = np.mean(reshaped, axis=0)
                seasonal_centers.append(np.mean(seasonal_pattern, axis=0))
            else:
                seasonal_centers.append(np.mean(cluster_points, axis=0))

        seasonal_centers = np.array(seasonal_centers)

        # Calculate DB scores
        season_scores = []
        
        for i in range(n_clusters):
            cluster_i = X[labels == unique_labels[i]]
            
            if len(cluster_i) <= 1:
                continue
                
            # Calculate within-cluster scatter
            season_scatter_i = np.std(cluster_i)
            max_season_ratio = 0
            
            for j in range(n_clusters):
                if i != j:
                    cluster_j = X[labels == unique_labels[j]]
                    if len(cluster_j) <= 1:
                        continue
                        
                    # Calculate between-cluster separation
                    season_sep = np.linalg.norm(seasonal_centers[i] - seasonal_centers[j])
                    season_scatter_j = np.std(cluster_j)
                    
                    if season_sep > 0:
                        season_ratio = (season_scatter_i + season_scatter_j) / season_sep
                        max_season_ratio = max(max_season_ratio, season_ratio)
            
            if max_season_ratio > 0:
                season_scores.append(max_season_ratio)

        # Return zeros for temporal and the seasonal score
        return np.mean(season_scores) if season_scores else 0

    # Convert DataFrame to numpy array if needed
    if isinstance(data_scaled, pd.DataFrame):
        data_scaled_array = data_scaled.values
    else:
        data_scaled_array = np.array(data_scaled)
    
    # Get standard DB score from sklearn
    standard_db = davies_bouldin_score(data_scaled_array, labels)
    
    # Get temporal and seasonal components
    seasonal_db = temporal_seasonal_scores(data_scaled_array, labels)
    
    # Combine scores (equal weights)
    combined_db = (standard_db + seasonal_db) / 2

    return {
        'davies_bouldin_combined': combined_db, 
        'davies_bouldin_value': standard_db,
        'davies_bouldin_seasonal': seasonal_db
    }

def cluster_Kmedoids_auto(grid, data, scaling_data=None, kmin=2, kmax=20, 
                         method='dynmsc', random_state=None, metric='euclidean', 
                         print_details=False):
    """
    Perform K-Medoids clustering with automatic cluster number selection using DynMSC.
    """
    from kmedoids import dynmsc
    
    data_scaled, scaler = _prepare_scaled_data(data, scaling_data)
    
    # Compute distance matrix if needed
    if metric != 'precomputed':
        from sklearn.metrics.pairwise import pairwise_distances
        dist_matrix = pairwise_distances(data_scaled, metric=metric)
    else:
        dist_matrix = data_scaled
    
    # Run DynMSC for automatic cluster selection
    result = dynmsc(dist_matrix, kmax, kmin)
    
    optimal_k = result.bestk
    
    if print_details:
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Medoid Silhouette scores: {result.losses}")
        print(f"Range of k tested: {result.rangek}")
    
    # Now run clustering with the optimal number of clusters
    return cluster_Kmedoids(grid, optimal_k, data, scaling_data, 
                          method='fasterpam', random_state=random_state, 
                          metric=metric, print_details=print_details)

def compare_kmedoids_methods(grid, n_clusters, data, scaling_data=None, 
                           methods=['fasterpam', 'fastpam1', 'pam', 'alternate'],
                           metric='euclidean', print_details=False):
    """
    Compare different K-Medoids methods.
    """
    results = {}
    
    for method in methods:
        try:
            start_time = time.time()
            method_results = cluster_Kmedoids(grid, n_clusters, data, scaling_data, 
                                            method=method, metric=metric, 
                                            print_details=False)
            time_taken = time.time() - start_time
            
            results[method] = {
                'CoV': method_results[1][0],
                'inertia': method_results[1][1],
                'time': time_taken
            }
        except Exception as e:
            print(f"Error with method {method}: {e}")
            results[method] = None
    
    if print_details:
        print("\nK-Medoids Methods Comparison:")
        print("=" * 50)
        for method, metrics in results.items():
            if metrics is not None:
                print(f"\n{method.upper()}:")
                print(f"  Coefficient of Variation: {metrics['CoV']:.4f}")
                print(f"  Inertia: {metrics['inertia']:.4f}")
                print(f"  Time: {metrics['time']:.3f}s")
    
    return results



