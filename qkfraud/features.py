import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standardized(X_train, tests):
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), {k: scaler.transform(v) for k, v in tests.items()}


def pca_angles(X_train, tests, d):
    """Standardize, project to ``d`` principal components and min-max scale to [0, pi].

    Every transformation is fitted on the training split; test values are clipped.
    Returns the transformed training set, the transformed test sets and the
    fraction of variance retained by the PCA.
    """
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=d, svd_solver="full").fit(scaler.transform(X_train))
    minmax = MinMaxScaler(feature_range=(0.0, np.pi)).fit(pca.transform(scaler.transform(X_train)))

    def transform(X):
        return np.clip(minmax.transform(pca.transform(scaler.transform(X))), 0.0, np.pi)

    return (
        transform(X_train),
        {k: transform(v) for k, v in tests.items()},
        float(pca.explained_variance_ratio_.sum()),
    )


def pca_cumulative_variance(X_train, d_max):
    pca = PCA(n_components=d_max, svd_solver="full").fit(StandardScaler().fit_transform(X_train))
    return np.cumsum(pca.explained_variance_ratio_)
