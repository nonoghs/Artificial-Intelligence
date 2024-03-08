from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)

    return x
    raise NotImplementedError


def get_covariance(dataset):
    covariance = (1/(len(dataset) - 1)) * np.dot(np.transpose(dataset), dataset)

    return covariance
    raise NotImplementedError


def get_eig(S, m):
    evalue, evector = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    diagonal = np.diag(evalue)

    return np.flip(diagonal), np.flip(evector, axis=1)
    raise NotImplementedError


def get_eig_prop(S, prop):
    """
    Get eigenvalues and eigenvectors of matrix S that explain more than 'prop'
    proportion of the variance.

    Parameters:
        S (numpy.ndarray): The input matrix, assumed to be symmetric.
        prop (float): Proportion of variance to be explained.

    Returns:
        Lambda (numpy.ndarray): Diagonal matrix of selected eigenvalues in descending order.
        U (numpy.ndarray): Matrix of corresponding eigenvectors in columns.
    """
    # Ensure S is a numpy array
    S = np.array(S)

    # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(S)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate the cumulative variance explained by the eigenvalues
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance

    # Identify eigenvalues that explain more than 'prop' proportion of the variance
    mask = explained_variance_ratio >= prop

    # Select the desired eigenvalues and eigenvectors
    selected_eigenvalues = eigenvalues[mask]
    selected_eigenvectors = eigenvectors[:, mask]

    # Create a diagonal matrix from the selected eigenvalues
    Lambda = np.diag(selected_eigenvalues)

    # Ensure the eigenvectors are arranged in columns corresponding to the eigenvalues
    U = selected_eigenvectors

    return Lambda, U
    raise NotImplementedError


def project_image(img, U):
    sum = 0
    for vector in np.transpose(U):
        alpha = np.dot(np.transpose(vector), img)
        sum += np.dot(alpha, vector)
    return sum
    raise NotImplementedError

def display_image(orig, proj):
    # reshape images
    reshape_orig = np.transpose(orig.reshape((32, 32)))
    reshape_proj = np.transpose(proj.reshape((32, 32)))
    # creating figure
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)
    ax1.set_title('Original')
    ax2.set_title('Projection')

    # adding colorbars
    axes_orig = ax1.imshow(reshape_orig, aspect='equal')
    orig_cbar = fig.colorbar(axes_orig, ax=ax1)

    axes_proj = ax2.imshow(reshape_proj, aspect='equal')
    proj_cbar = fig.colorbar(axes_proj, ax=ax2)

    return fig, ax1, ax2
    raise NotImplementedError