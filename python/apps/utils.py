# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import numpy.typing as npt
from scipy.cluster.vq import vq, kmeans2
from typing import Tuple
from time import perf_counter


def get_bin_metadata(bin_file) -> Tuple[int, int]:
    array = np.fromfile(file=bin_file, dtype=np.uint32, count=2)
    return array[0], array[1]


def bin_to_numpy(dtype, bin_file) -> np.ndarray:
    npts, ndims = get_bin_metadata(bin_file)
    return np.fromfile(file=bin_file, dtype=dtype, offset=8).reshape(npts, ndims)

class timer:
    last = perf_counter()

    def elapsed(self, round_digit:int = 3):
        new = perf_counter()
        elapsed_time = new - self.last
        self.last = new
        return round(elapsed_time, round_digit)


def numpy_to_bin(array, out_file):
    npts, ndims = np.shape(array)
    f = open(out_file, "wb")
    f.write(npts.to_bytes(4, "little"))
    f.write(ndims.to_bytes(4, "little"))
    f.write(array.tobytes())
    f.close()

def bin_to_sample(dtype, in_file, out_file, frac=0.01):
    npts, ndims = get_bin_metadata(in_file)
    array = np.memmap(in_file, dtype=dtype, mode='r', offset=8).reshape(npts, ndims)
    nsample = int(npts * frac)
    random_ids = np.random.choice(npts, nsample, replace=False)
    sub_array = array[random_ids]
    with open(out_file, "wb") as f:
        f.write(nsample.to_bytes(4, "little"))
        f.write(ndims.tobytes())
        f.write(sub_array.tobytes())
        f.close()        

def read_gt_file(gt_file) -> Tuple[npt.NDArray[int], npt.NDArray[float]]:
    """
    Return ids and distances to queries
    """
    nq, K = get_bin_metadata(gt_file)
    ids = np.fromfile(file=gt_file, dtype=np.uint32, offset=8, count=nq * K).reshape(
        nq, K
    )
    # dists = np.fromfile(
    #     file=gt_file, dtype=np.float32, offset=8 + nq * K * 4, count=nq * K
    # ).reshape(nq, K)
    dists = None
    return ids, dists


def calculate_recall(
    result_set_indices: npt.NDArray[int],
    truth_set_indices: npt.NDArray[int],
    recall_at: int = 5,
) -> float:
    """
    result_set_indices and truth_set_indices correspond by row index. the columns in each row contain the indices of
    the nearest neighbors, with result_set_indices being the approximate nearest neighbor results and truth_set_indices
    being the brute force nearest neighbor calculation via sklearn's NearestNeighbor class.
    :param result_set_indices:
    :param truth_set_indices:
    :param recall_at:
    :return:
    """
    found = 0
    for i in range(0, result_set_indices.shape[0]):
        result_set_set = set(result_set_indices[i][0:recall_at])
        truth_set_set = set(truth_set_indices[i][0:recall_at])
        found += len(result_set_set.intersection(truth_set_set))
    return found / (result_set_indices.shape[0] * recall_at)


def calculate_recall_from_gt_file(K: int, ids: npt.NDArray[int], gt_file: str) -> float:
    """
    Calculate recall from ids returned from search and those read from file
    """
    gt_ids, gt_dists = read_gt_file(gt_file)
    return calculate_recall(ids, gt_ids, K)


def cluster_and_permute(
    dtype_str, npts, ndims, data, num_clusters
) -> Tuple[npt.NDArray[int], npt.NDArray[int]]:
    """
    Cluster the data and return permutation of row indices
    that would group indices of the same cluster together
    """
    sample_size = min(100000, npts)
    sample_indices = np.random.choice(range(npts), size=sample_size, replace=False)
    sampled_data = data[sample_indices, :]
    centroids, sample_labels = kmeans2(sampled_data, num_clusters, minit="++", iter=10)
    labels, dist = vq(data, centroids)

    count = np.zeros(num_clusters)
    for i in range(npts):
        count[labels[i]] += 1
    print("Cluster counts")
    print(count)

    offsets = np.zeros(num_clusters + 1, dtype=int)
    for i in range(0, num_clusters, 1):
        offsets[i + 1] = offsets[i] + count[i]

    permutation = np.zeros(npts, dtype=int)
    counters = np.zeros(num_clusters, dtype=int)
    for i in range(npts):
        label = labels[i]
        row = offsets[label] + counters[label]
        counters[label] += 1
        permutation[row] = i

    return offsets, permutation
