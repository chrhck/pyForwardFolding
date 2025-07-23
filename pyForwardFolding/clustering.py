from typing import Dict, List

import numpy as np
from sklearn.cluster import HDBSCAN, MiniBatchKMeans


def compress_hdbscan(
    data: Dict[str, np.ndarray], weight_keys: List[str], min_cluster_size: int = 5
) -> Dict[str, np.ndarray]:
    # Use all keys that are not weight keys as observables
    observables = [key for key in data.keys() if key not in weight_keys]
    X = np.vstack([data[obs] for obs in observables]).T
    hdb = HDBSCAN(min_cluster_size=min_cluster_size)
    hdb.fit(X)

    max_label = np.max(hdb.labels_)
    # relabel noise points so that each noise point is in its own cluster

    n_noise = np.sum(hdb.labels_ == -1)

    cluster_labels = np.array(hdb.labels_, copy=True)
    cluster_labels[cluster_labels == -1] = np.arange(
        max_label + 1, max_label + 1 + n_noise
    )

    unique_labels = np.unique(cluster_labels)

    compressed = {dkey: np.zeros(unique_labels.shape[0]) for dkey in data.keys()}

    for label in unique_labels:
        label_mask = cluster_labels == label
        for dkey, dval in data.items():
            if dkey in weight_keys:
                compressed[dkey][label] = np.sum(dval[label_mask])
            else:
                compressed[dkey][label] = np.average(dval[label_mask])

    return compressed


def compress_minibatch_kmeans(
    data: Dict[str, np.ndarray], weight_keys: List[str], compression_factor: int = 10
) -> Dict[str, np.ndarray]:
    n_events = len(next(iter(data.values())))

    # Handle empty data
    if n_events == 0:
        return {dkey: np.array([]) for dkey in data.keys()}

    n_clusters = max(1, int(np.ceil(n_events / compression_factor)))

    X = np.vstack([data[key] for key in data.keys() if key not in weight_keys]).T
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    compressed = {dkey: np.zeros(n_clusters) for dkey in data.keys()}

    key_ix = 0
    for dkey in data.keys():
        if dkey in weight_keys:
            compressed[dkey] = np.bincount(
                cluster_labels, weights=data[dkey], minlength=n_clusters
            )
        else:
            compressed[dkey] = cluster_centers[:, key_ix]
            key_ix += 1

    return compressed
