# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
from xml.dom.pulldom import default_bufsize

import diskannpy
import numpy as np
import utils
import multiprocessing as mp
import faiss
import time

def ivf_build_and_search(data_file, query_file, out_file, K, vector_dtype):
    data = utils.bin_to_numpy(vector_dtype, data_file)
    query = utils.bin_to_numpy(vector_dtype, query_file)
    
    nd, d = data.shape
    nq = query.shape[0]
    nlist = int(10 * np.sqrt(nd))
    nprobe = 100

    index = faiss.index_factory(d, f"IVF{nlist},Flat")
    # res = faiss.StandardGpuResources()
    # index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # build ivf index
    index.train(data)
    index.add(data)

    # search ivf index
    index.nprobe = nprobe
    I = []
    nchunks = 100
    chunk_size = nq // nchunks
    for i in range(nchunks):
        start = i * chunk_size
        end =  nq if i == nchunks-1 else start + chunk_size
        dists, ids = index.search(query[start : end], K)
        I.append(ids)
    
    I = np.concatenate(I, axis=0)
    utils.numpy_to_bin(I, out_file)


def build_and_search(
    metric,
    dtype_str,
    index_directory,
    indexdata_file,
    querydata_file,
    Lb,
    graph_degree,
    K,
    Ls,
    ood_build,
    ood_lambda,
    max_nq_per_node,
    pre_knn,
    num_threads,
    gt_file,
    index_prefix,
    search_only
):
    if dtype_str == "float":
        dtype = np.single
    elif dtype_str == "int8":
        dtype = np.byte
    elif dtype_str == "uint8":
        dtype = np.ubyte
    else:
        raise ValueError("data_type must be float, int8 or uint8")

    # build index
    if not search_only:
        query_sample_file = ""
        qids_sample_file = ""
        if ood_build and max_nq_per_node > 0 and ood_lambda > 0:
            # create a query sample
            query_file_prefix = querydata_file.split('.')[0]
            query_sample_file = query_file_prefix + "_sample_data.bin"
            qids_sample_file = query_file_prefix + "_sample_nnids.bin"
            utils.bin_to_sample(dtype=dtype, in_file=querydata_file, out_file=query_sample_file, frac=0.1)
            ivf_build_and_search(
                                data_file=indexdata_file,
                                query_file=query_sample_file,
                                out_file=qids_sample_file,
                                K=pre_knn,
                                vector_dtype=dtype)
        
        diskannpy.build_memory_index(
            data=indexdata_file,
            metric=metric,
            vector_dtype=dtype,
            index_directory=index_directory,
            complexity=Lb,
            graph_degree=graph_degree,
            num_threads=num_threads,
            index_prefix=index_prefix,
            alpha=1.2,
            use_pq_build=False,
            num_pq_bytes=8,
            use_opq=False,
            query_sample=query_sample_file,
            qids_sample=qids_sample_file,
            ood_lambda=ood_lambda,
            max_nq_per_node=max_nq_per_node,
            ood_build=ood_build,
        )

    # ready search object
    index = diskannpy.StaticMemoryIndex(
        metric=metric,
        vector_dtype=dtype,
        data_path=indexdata_file,
        index_directory=index_directory,
        num_threads=num_threads,  # this can be different at search time if you would like
        initial_search_complexity=Ls,
        index_prefix=index_prefix
    )

    queries = utils.bin_to_numpy(dtype, querydata_file)

    timer = utils.timer()
    num_bins = 16
    num_labels = 131073
    chunk_size = (num_labels + num_bins - 1) // num_bins
    start = time.time()
    # ids, dists = index.batch_search_binned(queries, K, Ls, num_bins, chunk_size, num_threads)
    ids, dists = index.batch_search(queries, K, Ls, num_threads)
    print(time.time() - start)
    # query_time = timer.elapsed()
    # qps = round(queries.shape[0]/query_time, 1)
    # print('Batch searched', queries.shape[0], 'in', query_time, 's @', qps, 'QPS')
    
    # indices = []
    # with mp.Pool(num_bins) as p:
    #     indices = p.starmap(filter, map(lambda i : (ids, i, chunk_size, K // num_bins), range(num_bins)))

    # ids = np.stack(indices, axis=0)
    
    print(ids.shape)
    # for i in range(num_bins):
    #     ids[i] = ids[i] + i * chunk_size
    # ids = ids.transpose(1, 0, 2).reshape(-1, K)

    if gt_file != "":
        recall = utils.calculate_recall_from_gt_file(K, ids, gt_file)
        print(f"recall@{K} is {recall}")
    
    return ids

def filter(ann_indices, i, numy_per_gpu, topk):
    start_label = i * numy_per_gpu
    end_label = (i+1) * numy_per_gpu
    mask = (ann_indices >= start_label) & (ann_indices < end_label)
    ann_indices = ann_indices * mask
    indices = np.argpartition(mask, kth=-topk, axis=-1)[:, -topk:]
    filtered_ann_indices = ann_indices[np.arange(indices.shape[0])[:, None], indices]
    filtered_ann_indices = filtered_ann_indices % numy_per_gpu
    return filtered_ann_indices

def sharded_build_and_search(
    metric,
    dtype_str,
    index_directory,
    indexdata_file,
    querydata_file,
    Lb,
    graph_degree,
    K,
    Ls,
    num_shards,
    num_threads,
    gt_file,
    index_prefix,
    search_only
):
    if dtype_str == "float":
        dtype = np.single
    elif dtype_str == "int8":
        dtype = np.byte
    elif dtype_str == "uint8":
        dtype = np.ubyte
    else:
        raise ValueError("data_type must be float, int8 or uint8")

    # divide data into shards
    data = utils.bin_to_numpy(dtype, indexdata_file)
    npts, ndims = utils.get_bin_metadata(indexdata_file)
    chunk_size = (npts + num_shards - 1) // num_shards
    K_shard = K // num_shards
    ids = [] 
    total_time = 0
    for i in range(num_shards):
        start = i * chunk_size
        end = (i+1) * chunk_size
        data_i = data[start : end]
        chunk_datafile = indexdata_file.split('.')[0] + '_' + str(i) + '.bin'
        utils.numpy_to_bin(data_i, chunk_datafile)
        start = time.time()
        curr_ids = build_and_search(metric, dtype_str, index_directory, 
                                    chunk_datafile, querydata_file, 
                                    Lb, graph_degree, K_shard, Ls, 
                                    False, 1., 0, 0, 
                                    num_threads, "", index_prefix + "_"  + str(i), search_only)
        curr_ids += i * chunk_size
        ids.append(curr_ids)
        total_time += time.time() - start
    
    ids = np.concatenate(ids, axis=1)
    print(ids.shape)

    if gt_file != "":
        recall = utils.calculate_recall_from_gt_file(K, ids, gt_file)
        print(f"recall@{K} is {recall}")
    print("time taken: ", total_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="in-mem-static",
        description="Static in-memory build and search from vectors in a file",
    )

    parser.add_argument("-m", "--metric", required=False, default="l2")
    parser.add_argument("-d", "--data_type", required=True)
    parser.add_argument("-id", "--index_directory", required=False, default=".")
    parser.add_argument("-i", "--indexdata_file", required=True)
    parser.add_argument("-q", "--querydata_file", required=True)
    parser.add_argument("-Lb", "--Lbuild", default=50, type=int)
    parser.add_argument("-Ls", "--Lsearch", default=50, type=int)
    parser.add_argument("-R", "--graph_degree", default=32, type=int)
    parser.add_argument("--ood_build", required=False, default=False, type=bool)
    parser.add_argument("--max_nq_per_node", default=5, type=int)
    parser.add_argument("--ood_lambda", default=0.75, type=float)
    parser.add_argument("--pre_knn", default=100, type=int)
    parser.add_argument("-S", "--num_shards", default=16, type=int)
    parser.add_argument("-T", "--num_threads", default=8, type=int)
    parser.add_argument("-K", default=10, type=int)
    parser.add_argument("-G", "--gt_file", default="")
    parser.add_argument("-ip", "--index_prefix", required=False, default="ann")
    parser.add_argument("--search_only", required=False, default=False)
    args = parser.parse_args()

    # build_and_search(
    #     args.metric,
    #     args.data_type,
    #     args.index_directory.strip(),
    #     args.indexdata_file.strip(),
    #     args.querydata_file.strip(),
    #     args.Lbuild,
    #     args.graph_degree,  # Build args
    #     args.K,
    #     args.Lsearch,
    #     args.ood_build,
    #     args.ood_lambda,
    #     args.max_nq_per_node,
    #     args.pre_knn,
    #     args.num_threads,  # search args
    #     args.gt_file,
    #     args.index_prefix,
    #     args.search_only
    # )

    sharded_build_and_search(
        args.metric,
        args.data_type,
        args.index_directory.strip(),
        args.indexdata_file.strip(),
        args.querydata_file.strip(),
        args.Lbuild,
        args.graph_degree,  # Build args
        args.K,
        args.Lsearch,
        args.num_shards,
        args.num_threads,  # search args
        args.gt_file,
        args.index_prefix,
        args.search_only
    )
