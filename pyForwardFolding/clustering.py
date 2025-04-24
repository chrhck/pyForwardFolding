import numpy as np
from sklearn.cluster import HDBSCAN


def compress_hdbscan(data, observables, weight_keys, min_cluster_size=5):
    X = np.vstack([data[obs] for obs in observables]).T
    hdb = HDBSCAN(min_cluster_size=min_cluster_size)
    hdb.fit(X)

    max_label = np.max(hdb.labels_)
    # relabel noise points so that each noise point is in its own cluster

    n_noise = np.sum(hdb.labels_ == -1)

    cluster_labels = np.array(hdb.labels_, copy=True)
    cluster_labels[cluster_labels == -1] = np.arange(max_label + 1, max_label + 1 + n_noise)

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

