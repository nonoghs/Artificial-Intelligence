import csv
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    """
    Load data from a CSV file and return it as a list of dictionaries.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        List[Dict]: List of dictionaries representing the data.
    """
    data = []
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as file:
            # Use DictReader to directly create dictionaries from the rows
            reader = csv.DictReader(file)
            for row in reader:
                # Ensure all rows are plain dictionaries, not OrderedDict
                data.append(dict(row))
        return data
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def calc_features(row):
    """
    Calculate the feature vector for a country.

    Parameters:
        row (Dict): Data for one country.

    Returns:
        np.ndarray: NumPy array of shape (6,) containing the feature vector.
    """
    try:
        # Extract and convert the relevant data
        x1 = float(row['Population'])
        x2 = float(row['Net migration'])
        x3 = float(row['GDP ($ per capita)'])
        x4 = float(row['Literacy (%)'])
        x5 = float(row['Phones (per 1000)'])
        x6 = float(row['Infant mortality (per 1000 births)'])

        # Create and return the feature vector
        return np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)
    except KeyError as e:
        print(f"Missing expected column: {str(e)}")
        return None
    except ValueError as e:
        print(f"Cannot convert to float: {str(e)}")
        return None


def hac(features):
    n = len(features)
    Matrix = np.zeros((n, n))
    result = np.zeros((n - 1, 4))
    index = -1
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                Matrix[i][j] = sys.maxsize
            else:
                Matrix[i][j] = np.linalg.norm(np.array(features[i]) - np.array(features[j]))

    cluster_index = np.array([index] * (n))
    for i in range(0, n - 1):
        Distance_min = np.amin(Matrix)
        index_Small_min = sys.maxsize
        index_Big_min = sys.maxsize
        for row_j in range(n):
            for column_k in range(n):
                if Matrix[row_j][column_k] == Distance_min:
                    if cluster_index[row_j] == -1:
                        answer1 = row_j
                    else:
                        answer1 = cluster_index[row_j] + n
                    if cluster_index[column_k] == -1:
                        answer2 = column_k
                    else:
                        answer2 = cluster_index[column_k] + n
                    if answer1 > answer2:
                        answer1_big = True
                    else:
                        answer1_big = False
                    index_Small = min(answer1, answer2)
                    index_Big = max(answer1, answer2)
                    # update smallest index
                    if index_Small < index_Small_min:
                        index_Small_min = index_Small
                        index_Big_min = index_Big
                        if answer1_big:
                            row = row_j
                            column = column_k
                        else:
                            row = column_k
                            column = row_j
                    elif index_Small == index_Big_min and index_Big < index_Big_min:
                        index_Small_min = index_Small
                        index_Big_min = index_Big
                        if answer1_big:
                            row = row_j
                            column = column_k
                        else:
                            row = column_k
                            column = row_j

        cluster_for_row = list()
        cluster_for_column = list()
        if cluster_index[row] == -1:
            cluster_for_row.append(row)
        else:
            cluster_for_row = [j for j in range(len(cluster_index)) if cluster_index[j] == cluster_index[row]]
        if cluster_index[column] == -1:
            cluster_for_column.append(column)
        else:
            cluster_for_column = [j for j in range(len(cluster_index)) if cluster_index[j] == cluster_index[column]]

        cluster_merge = cluster_for_column + cluster_for_row
        result[i][0] = index_Small_min
        result[i][1] = index_Big_min
        result[i][2] = Distance_min
        result[i][3] = len(cluster_merge)

        for col in cluster_for_column:
            for r in cluster_for_row:
                Matrix[r][col] = sys.maxsize
        for j in range(0, n):
            if j not in (cluster_merge):
                Matrix[j][cluster_merge] = max(Matrix[j][k] for k in cluster_merge)
        for j in cluster_merge:
            for z in range(0, n):
                Matrix[j][z] = Matrix[z][j]
        # update in cluster data
        cluster_index[cluster_merge] = i
    return result


def fig_hac(Z, names):
    """
    Visualize the hierarchical agglomerative clustering.

    Parameters:
        Z (np.ndarray): NumPy array representing the hierarchical clustering.
        names (List[str]): List of strings representing country names.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure object.
    """
    # Initialize figure
    fig = plt.figure(figsize=(10, 6))

    # Create a dendrogram
    dendrogram(Z, labels=names, leaf_rotation=90)

    # Adjust plot to make sure labels are visible
    plt.tight_layout()

    # Show the plot
    plt.show()

    return fig


def normalize_features(features):
    """
    Normalize the feature vectors.

    Parameters:
        features (List[np.ndarray]): List of NumPy arrays, each of shape (6,) and dtype float64.

    Returns:
        List[np.ndarray]: List of normalized NumPy arrays, each of shape (6,) and dtype float64.
    """
    # Convert the list of 1D arrays into a 2D array
    feature_matrix = np.array(features)

    # Calculate the mean and standard deviation for each feature
    means = np.mean(feature_matrix, axis=0)
    std_devs = np.std(feature_matrix, axis=0)

    # Normalize the features
    normalized_features = (feature_matrix - means) / std_devs

    return normalized_features.tolist()  # Convert back to a list of arrays


if __name__ == "__main__":
    # Load data from CSV file
    data = load_data("countries.csv")

    # Extract country names and features from data
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]

    # Normalize features
    features_normalized = normalize_features(features)

    # Test with various values of n
    n = 20  # Change this value to test with different numbers of countries

    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])

    # Visualize the results
    fig = fig_hac(Z_raw, country_names[:n])
    plt.show()

    fig = fig_hac(Z_normalized, country_names[:n])
    plt.show()






