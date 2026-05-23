"""
Dataset complexity computation.

Pipeline: embeddings (N, M) -> distance matrix -> kernel matrix -> C(X) = log det(I + K).

Each stage is independent and swappable.

The embedding step (score-based, trajectory-based, etc.) happens upstream
and is not part of this module. This module operates on generic embeddings.

Bandwidth selection: lambda must NOT depend on the dataset being measured.
It is calibrated from an independent reference sample drawn from the model's
training distribution, making it a property of the score model.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


# ============================================================
# Pairwise distances
# ============================================================

def pairwise_distance(embeddings, metric='euclidean'):
    """
    Compute pairwise distance matrix.

    Args:
        embeddings: (N, M) array
        metric: any metric accepted by scipy.spatial.distance.pdist

    Returns:
        D: (N, N) symmetric distance matrix
    """
    dists = pdist(embeddings, metric=metric)
    return squareform(dists)


# ============================================================
# Bandwidth calibration
# ============================================================

def calibrate_bandwidth(reference_embeddings, metric='euclidean'):
    """
    Calibrate bandwidth from an independent reference sample.

    Uses the median heuristic on embeddings from a reference sample drawn
    from the model's training distribution. This makes lambda a property
    of the score model and data distribution, not of the specific dataset
    being measured.

    Args:
        reference_embeddings: (N_ref, M) embeddings from a representative
            sample of the training distribution

    Returns:
        lambda_: scalar bandwidth, calibrated once and reused for all
            complexity evaluations under this model
    """
    D = pairwise_distance(reference_embeddings, metric=metric)
    return median_heuristic(D)


def median_heuristic(D):
    """
    Bandwidth via median heuristic on a distance matrix.

    WARNING: Do not use this on the dataset being measured — that makes
    lambda depend on the input and defeats the purpose of the complexity
    measure. Use calibrate_bandwidth() with an independent reference sample.

    Args:
        D: (N, N) distance matrix

    Returns:
        lambda_: scalar bandwidth
    """
    upper = D[np.triu_indices_from(D, k=1)]
    if len(upper) == 0:
        return 1.0
    med_sq = np.median(upper ** 2)
    if med_sq == 0:
        return 1.0
    return 1.0 / med_sq


# ============================================================
# Kernel functions
# ============================================================

def gaussian_kernel(D, lambda_):
    """
    Gaussian (RBF) kernel: K_{ij} = exp(-lambda * D_{ij}^2).

    Args:
        D: (N, N) distance matrix
        lambda_: bandwidth parameter (> 0)

    Returns:
        K: (N, N) PSD kernel matrix
    """
    return np.exp(-lambda_ * D ** 2)


def laplacian_kernel(D, lambda_):
    """
    Laplacian kernel: K_{ij} = exp(-sqrt(lambda) * D_{ij}).

    Args:
        D: (N, N) distance matrix
        lambda_: bandwidth parameter (> 0)

    Returns:
        K: (N, N) PSD kernel matrix
    """
    return np.exp(-np.sqrt(lambda_) * D)


# ============================================================
# Complexity
# ============================================================

def complexity(K):
    """
    Compute C(X) = log det(I + K) via Cholesky factorization.

    Also returns eigenvalues of K for analysis.

    Args:
        K: (N, N) PSD kernel matrix

    Returns:
        C: scalar — the complexity value
        eigenvalues: (N,) array — eigenvalues of K in decreasing order
    """
    N = K.shape[0]

    # Eigenvalues for analysis
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.clip(eigenvalues, 0, None)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Log-det via Cholesky (I + K is guaranteed PD)
    L = np.linalg.cholesky(np.eye(N) + K)
    C = 2.0 * np.sum(np.log(np.diag(L)))

    return C, eigenvalues


# ============================================================
# Full pipeline
# ============================================================

def compute_complexity(embeddings, lambda_, kernel_fn=gaussian_kernel):
    """
    Full pipeline: embeddings -> complexity.

    Lambda must be provided explicitly. Use calibrate_bandwidth() with an
    independent reference sample to obtain it.

    Args:
        embeddings: (N, M) array — pre-computed embeddings (score, trajectory, etc.)
        lambda_: kernel bandwidth (required). Obtain via calibrate_bandwidth()
            on an independent reference sample.
        kernel_fn: callable(D, lambda_) -> K. Default: gaussian_kernel.

    Returns:
        result: dict with keys:
            'C': scalar complexity
            'K': (N, N) kernel matrix
            'D': (N, N) distance matrix
            'eigenvalues': (N,) eigenvalues of K
            'lambda': bandwidth used
    """
    D = pairwise_distance(embeddings)

    K = kernel_fn(D, lambda_)
    C, eigenvalues = complexity(K)

    return {
        'C': C,
        'K': K,
        'D': D,
        'eigenvalues': eigenvalues,
        'lambda': lambda_,
    }


# ============================================================
# Legacy: score embedding helper
# ============================================================

def score_embedding(score_vectors):
    """
    Flatten score vectors into embedding vectors.

    Args:
        score_vectors: (N, L, D) array — scores at L noise levels, D dimensions

    Returns:
        embeddings: (N, D*L) array — concatenated score vectors
    """
    N = score_vectors.shape[0]
    return score_vectors.reshape(N, -1)
