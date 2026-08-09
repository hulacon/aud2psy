"""Unified dimensionality reduction projections.

Provides a single interface for PCA, PPCA, UMAP, t-SNE, and MDS projections
used across scatter, explorer, and other visualization modules.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _ppca_em(
    X: np.ndarray,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Probabilistic PCA via EM algorithm.

    Handles missing data (NaN values) naturally through the EM framework.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features). May contain NaN values.
    n_components : int
        Number of latent dimensions.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance for log-likelihood.
    random_state : int
        Random seed for initialization.

    Returns
    -------
    Z : np.ndarray
        Latent coordinates of shape (n_samples, n_components).
    W : np.ndarray
        Loading matrix of shape (n_features, n_components).
    sigma2 : float
        Noise variance estimate.
    var_explained : np.ndarray
        Approximate variance explained by each component.
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    # Identify missing values
    missing_mask = np.isnan(X)
    has_missing = np.any(missing_mask)

    # Initialize with column means for missing values (for initial stats)
    X_filled = X.copy()
    if has_missing:
        col_means = np.nanmean(X, axis=0)
        for j in range(n_features):
            X_filled[missing_mask[:, j], j] = col_means[j]

    # Center the data
    mu = np.nanmean(X, axis=0)
    X_centered = X_filled - mu

    # Initialize parameters
    # W: loading matrix (n_features x n_components)
    # sigma2: noise variance
    W = rng.randn(n_features, n_components) * 0.1
    sigma2 = 1.0

    prev_ll = -np.inf

    for iteration in range(max_iter):
        # E-step: compute expected latent variables
        # M = W'W + sigma2*I
        M = W.T @ W + sigma2 * np.eye(n_components)
        M_inv = np.linalg.inv(M)

        # For each sample, compute E[z|x] and E[zz'|x]
        Z = np.zeros((n_samples, n_components))
        Ez_zzT = np.zeros((n_samples, n_components, n_components))

        for i in range(n_samples):
            if has_missing and np.any(missing_mask[i]):
                # Handle missing data for this sample
                obs_idx = ~missing_mask[i]
                W_obs = W[obs_idx, :]
                x_obs = X_centered[i, obs_idx]

                M_obs = W_obs.T @ W_obs + sigma2 * np.eye(n_components)
                M_obs_inv = np.linalg.inv(M_obs)

                Z[i] = M_obs_inv @ W_obs.T @ x_obs
                Ez_zzT[i] = sigma2 * M_obs_inv + np.outer(Z[i], Z[i])
            else:
                Z[i] = M_inv @ W.T @ X_centered[i]
                Ez_zzT[i] = sigma2 * M_inv + np.outer(Z[i], Z[i])

        # M-step: update W and sigma2
        # W_new = (sum_i x_i E[z_i]') (sum_i E[z_i z_i'])^{-1}
        sum_xz = np.zeros((n_features, n_components))
        sum_zzT = np.zeros((n_components, n_components))

        for i in range(n_samples):
            if has_missing and np.any(missing_mask[i]):
                obs_idx = ~missing_mask[i]
                sum_xz[obs_idx] += np.outer(X_centered[i, obs_idx], Z[i])
            else:
                sum_xz += np.outer(X_centered[i], Z[i])
            sum_zzT += Ez_zzT[i]

        W_new = sum_xz @ np.linalg.inv(sum_zzT)

        # Update sigma2
        sigma2_new = 0.0
        n_obs = 0
        for i in range(n_samples):
            if has_missing:
                obs_idx = ~missing_mask[i]
                x_obs = X_centered[i, obs_idx]
                W_obs = W_new[obs_idx, :]
                recon = W_obs @ Z[i]
                sigma2_new += np.sum((x_obs - recon) ** 2)
                sigma2_new += sigma2 * np.trace(W_obs @ M_inv @ W_obs.T)
                n_obs += np.sum(obs_idx)
            else:
                recon = W_new @ Z[i]
                sigma2_new += np.sum((X_centered[i] - recon) ** 2)
                sigma2_new += sigma2 * np.trace(W_new @ M_inv @ W_new.T)
                n_obs += n_features

        sigma2_new /= n_obs
        sigma2_new = max(sigma2_new, 1e-6)  # Prevent collapse

        W = W_new
        sigma2 = sigma2_new

        # Compute log-likelihood for convergence check
        # Simplified: use reconstruction error as proxy
        ll = -0.5 * n_obs * np.log(sigma2)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # Compute approximate variance explained
    # Based on singular values of W
    _, s, _ = np.linalg.svd(W, full_matrices=False)
    var_components = s ** 2
    total_var = np.sum(var_components) + sigma2 * (n_features - n_components)
    var_explained = var_components / total_var

    return Z, W, sigma2, var_explained


def compute_projection(
    X: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """Compute dimensionality reduction projection.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    method : str
        Projection method: "pca", "ppca", "umap", "tsne", "mds", or "mds_nonmetric".
    n_components : int
        Number of output dimensions (default: 2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_proj : np.ndarray
        Projected coordinates of shape (n_samples, n_components).
    info : dict
        Additional info including axis labels.
        Keys: "xlabel", "ylabel", "method", "n_features", "n_samples"

    Raises
    ------
    ValueError
        If method is unknown or data is insufficient.
    ImportError
        If optional dependency (umap-learn) is not installed.

    Notes
    -----
    PPCA (Probabilistic PCA) uses an EM algorithm that naturally handles
    missing data (NaN values). Other methods require complete data and will
    fill NaN values with column means.
    """
    n_samples, n_features = X.shape

    # Validate inputs
    if n_samples < 3:
        raise ValueError(f"Need at least 3 samples for projection, got {n_samples}.")

    if n_features < 2:
        raise ValueError(f"Need at least 2 features for projection, got {n_features}.")

    # Handle NaN values (PPCA handles them natively, others need imputation)
    has_nan = np.any(np.isnan(X))
    if has_nan and method != "ppca":
        warnings.warn("NaN values found, filling with column means. Consider using PPCA for native missing data handling.")
        col_means = np.nanmean(X, axis=0)
        for i in range(n_features):
            mask = np.isnan(X[:, i])
            X[mask, i] = col_means[i]

    # Standardize features (for PPCA with missing data, use nanmean/nanstd)
    if method == "ppca" and has_nan:
        # Manual standardization that preserves NaN
        mu = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1  # Prevent division by zero
        X_scaled = (X - mu) / std
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    info = {
        "method": method,
        "n_features": n_features,
        "n_samples": n_samples,
    }

    if method == "pca":
        projector = PCA(n_components=n_components, random_state=random_state)
        X_proj = projector.fit_transform(X_scaled)
        var_explained = projector.explained_variance_ratio_

        if n_components == 2:
            info["xlabel"] = f"PC1 ({var_explained[0]:.1%} var)"
            info["ylabel"] = f"PC2 ({var_explained[1]:.1%} var)"
        elif n_components == 3:
            info["xlabel"] = f"PC1 ({var_explained[0]:.1%})"
            info["ylabel"] = f"PC2 ({var_explained[1]:.1%})"
            info["zlabel"] = f"PC3 ({var_explained[2]:.1%})"
        info["variance_explained"] = var_explained.tolist()

    elif method == "ppca":
        # Probabilistic PCA via EM - handles missing data natively
        X_proj, W, sigma2, var_explained = _ppca_em(
            X_scaled, n_components=n_components, random_state=random_state
        )

        if n_components == 2:
            info["xlabel"] = f"PPCA1 ({var_explained[0]:.1%} var)"
            info["ylabel"] = f"PPCA2 ({var_explained[1]:.1%} var)"
        elif n_components == 3:
            info["xlabel"] = f"PPCA1 ({var_explained[0]:.1%})"
            info["ylabel"] = f"PPCA2 ({var_explained[1]:.1%})"
            info["zlabel"] = f"PPCA3 ({var_explained[2]:.1%})"
        info["variance_explained"] = var_explained.tolist()
        info["noise_variance"] = float(sigma2)
        if has_nan:
            info["missing_data_handled"] = True

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP projection. "
                "Install with: pip install umap-learn"
            )

        projector = umap.UMAP(n_components=n_components, random_state=random_state)
        X_proj = projector.fit_transform(X_scaled)

        if n_components == 2:
            info["xlabel"] = "UMAP1"
            info["ylabel"] = "UMAP2"
        elif n_components == 3:
            info["xlabel"] = "UMAP1"
            info["ylabel"] = "UMAP2"
            info["zlabel"] = "UMAP3"

    elif method == "tsne":
        from sklearn.manifold import TSNE

        # Adjust perplexity for small datasets
        perplexity = min(30, n_samples - 1)
        projector = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
        )
        X_proj = projector.fit_transform(X_scaled)

        if n_components == 2:
            info["xlabel"] = "t-SNE1"
            info["ylabel"] = "t-SNE2"
        elif n_components == 3:
            info["xlabel"] = "t-SNE1"
            info["ylabel"] = "t-SNE2"
            info["zlabel"] = "t-SNE3"
        info["perplexity"] = perplexity

    elif method == "mds":
        from sklearn.manifold import MDS

        # Warn about performance for large datasets
        if n_samples > 10000:
            warnings.warn(
                f"MDS with {n_samples} points may be slow. "
                "Consider using PCA or subsampling for faster results."
            )

        projector = MDS(
            n_components=n_components,
            metric_mds=True,
            random_state=random_state,
            normalized_stress="auto",
            n_init=1,
            init="random",
        )
        X_proj = projector.fit_transform(X_scaled)

        if n_components == 2:
            info["xlabel"] = "MDS1"
            info["ylabel"] = "MDS2"
        elif n_components == 3:
            info["xlabel"] = "MDS1"
            info["ylabel"] = "MDS2"
            info["zlabel"] = "MDS3"
        info["stress"] = projector.stress_

    elif method == "mds_nonmetric":
        from sklearn.manifold import MDS

        # Warn about performance for large datasets
        if n_samples > 10000:
            warnings.warn(
                f"Non-metric MDS with {n_samples} points may be slow. "
                "Consider using PCA or subsampling for faster results."
            )

        projector = MDS(
            n_components=n_components,
            metric_mds=False,  # Non-metric MDS
            random_state=random_state,
            normalized_stress="auto",
            n_init=1,
            init="random",
        )
        X_proj = projector.fit_transform(X_scaled)

        if n_components == 2:
            info["xlabel"] = "NMDS1"
            info["ylabel"] = "NMDS2"
        elif n_components == 3:
            info["xlabel"] = "NMDS1"
            info["ylabel"] = "NMDS2"
            info["zlabel"] = "NMDS3"
        info["stress"] = projector.stress_

    else:
        valid_methods = ["pca", "ppca", "umap", "tsne", "mds", "mds_nonmetric"]
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Valid methods: {', '.join(valid_methods)}"
        )

    return X_proj, info


# List of valid projection methods for CLI argument validation
PROJECTION_METHODS = ["pca", "ppca", "umap", "tsne", "mds", "mds_nonmetric"]
