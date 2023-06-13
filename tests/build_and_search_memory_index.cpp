// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"

namespace po = boost::program_options;

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
int build_and_search(const diskann::Metric &metric, const std::string &data_path, const uint32_t R,
                const uint32_t Lbuild, const float alpha, const uint32_t num_threads,
                const bool use_pq_build, const size_t num_pq_bytes, const bool use_opq,
                const std::string &query_file, const std::string &result_path_prefix, const std::string &gt_path_prefix,
                const uint32_t recall_at, const uint32_t Lsearch, const uint32_t num_bins, const bool use_adj_sample)
{
    diskann::IndexWriteParameters paras = diskann::IndexWriteParametersBuilder(Lbuild, R)
                                              .with_alpha(alpha)
                                              .with_saturate_graph(false)
                                              .with_num_threads(num_threads)
                                              .build();

    size_t data_num, data_dim;
    diskann::get_bin_metadata(data_path, data_num, data_dim);

    uint32_t search_list_size_init = std::max(Lbuild, std::max(recall_at, Lsearch));

    diskann::Index<T, TagT, LabelT> index(metric, data_dim, data_num, false, 
                                          paras, search_list_size_init, 0, false, false, use_pq_build, num_pq_bytes,
                                          use_opq);
    
    // build index
    auto s = std::chrono::high_resolution_clock::now();
    
    index.build(data_path.c_str(), data_num, paras);

    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    std::cout << "Indexing time: " << diff.count() << "\n";

    // search index
    T *query = nullptr;
    uint32_t *gt_ids = nullptr, *gt_indptr = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_id_num, gt_id_dim, gt_indptr_num, gt_indptr_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);


    if (use_adj_sample) 
    {
        bool found_gt_ids = false;
        if (gt_path_prefix != std::string("null")) 
        {
            auto gt_ids_file = gt_path_prefix + "_indices.bin";
            auto gt_indptr_file = gt_path_prefix + "_indptr.bin";
            if (file_exists(gt_ids_file) && file_exists(gt_indptr_file)) 
            {
                diskann::load_bin<uint32_t>(gt_ids_file, gt_ids, gt_id_num, gt_id_dim);
                diskann::load_bin<uint32_t>(gt_indptr_file, gt_indptr, gt_indptr_num, gt_indptr_dim);
            }
        }
    }

    std::vector<uint32_t> bins(num_bins - 1);
    auto num_chunks = (data_num + num_bins - 1) / num_bins;
    for (uint32_t i = 1; i < num_bins; i++)
        bins[i - 1] = num_chunks * i;
    auto K_bin = recall_at / num_bins;
    std::vector<std::vector<uint32_t>> query_result_ids_binned(num_bins);
    uint32_t padding_val = (uint32_t) data_num;
    for (auto &qids_binned : query_result_ids_binned)
        qids_binned.resize(K_bin * query_num, padding_val);
    
    std::random_device device;
    std::mt19937       generator(device());

    s = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)query_num; i++)
    {
        uint32_t query_result_ids[recall_at];
        if (use_adj_sample)
        {
            uint32_t start_id = gt_indptr[i], end_id = gt_indptr[i+1];
            auto len = end_id - start_id;
            std::vector<uint32_t> curr_ids(gt_ids + start_id, gt_ids + end_id);
            if (len >= Lsearch) 
            {
                std::vector<uint32_t> samples(Lsearch);
                std::uniform_int_distribution<> dis(0, len - 1);
                for (auto& x : samples)
                    x = curr_ids[dis(generator)];
                index.search_with_adj_lookup(query + i * query_aligned_dim,
                                                samples, recall_at,
                                                query_result_ids);
            }    
            else
            {
                index.search_with_adj_lookup(query + i * query_aligned_dim,
                                                curr_ids, recall_at,
                                                query_result_ids);
            }
        }
        else
        {
            index.search(query + i * query_aligned_dim, recall_at, Lsearch,
                            query_result_ids).second;
        }
        if (num_bins <= 1)
            continue;
        
        uint32_t start_id = i * K_bin;
        uint32_t chunk_pos[num_bins] = {start_id};

        for (uint32_t j = 0; j < recall_at; j++) 
        {
            auto id = query_result_ids[j];
            // search for the chunk to be binned to
            uint32_t chunk_id;
            if (id < bins[0])
                chunk_id = 0;
            else
            {
                auto it = std::upper_bound(bins.begin(), bins.end(), id);
                chunk_id = std::distance(bins.begin(), it);
            }

            auto& pos = chunk_pos[chunk_id];
            if (pos < start_id + K_bin)
            { 
                query_result_ids_binned[chunk_id][pos] = (chunk_id > 0) ? (id - bins[chunk_id - 1]) : id;
                pos++;
            }

        }
    }
    diff = std::chrono::high_resolution_clock::now() - s;
    std::cout << "Searching time: " << diff.count() << "\n";

    std::cout << "Done searching. Now saving results " << std::endl;
        for (uint32_t i = 0; i < num_bins; i++)
    {
        std::string cur_result_path = result_path_prefix + "_rank_" + std::to_string(i) + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids_binned[i].data(), query_num, K_bin);
    }

    diskann::aligned_free(query);

    return 0;
}

int main(int argc, char** argv)
{
    std::string data_type, dist_fn, data_path, query_path, result_path, gt_path_prefix;
    uint32_t num_threads, R, Lbuild, build_PQ_bytes, K, Lsearch, num_bins;
    float alpha;
    bool use_pq_build, use_opq, use_adj_sample;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/cosine>");
        desc.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                           "Input data file in bin format");
        desc.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64), "Maximum graph degree");
        desc.add_options()("Lbuild,L", po::value<uint32_t>(&Lbuild)->default_value(100),
                           "Build complexity, higher value results in better graphs");
        desc.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                           "alpha controls density and diameter of graph, set "
                           "1 for sparse graph, "
                           "1.2 or 1.4 for denser graphs with lower diameter");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads used for building index (defaults to "
                           "omp_get_num_procs())");
        desc.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
                           "Number of PQ bytes to build the index; 0 for full precision "
                           "build");
        desc.add_options()("use_opq", po::bool_switch()->default_value(false),
                           "Set true for OPQ compression while using PQ "
                           "distance comparisons for "
                           "building the index, and false for PQ compression");
        desc.add_options()("query_path", po::value<std::string>(&query_path)->required(),
                           "Query file in binary format");
        desc.add_options()("result_path", po::value<std::string>(&result_path)->required(),
                           "Path prefix for saving results of the queries");
        desc.add_options()("gt_path_prefix", po::value<std::string>(&gt_path_prefix)->default_value(std::string("null")),
                           "ground truth file path prefix for the queryset");
        desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(), "Number of neighbors to be returned");
        desc.add_options()("Lsearch", po::value<uint32_t>(&Lsearch)->required(),
                           "List length for search");
        desc.add_options()("num_bins", po::value<uint32_t>(&num_bins)->default_value(1), 
                           "Number of bins to divide the retrieved results into");
        desc.add_options()("adj_samp", po::value<bool>(&use_adj_sample)->default_value(false),
                           "Whether to use adjacency list based sampling or not");
        
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    assert (num_bins > 0);

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    try
    {
        if (data_type == std::string("int8"))
        {
            return build_and_search<int8_t>(metric, data_path, R, Lbuild, alpha, num_threads,
                                            use_pq_build, build_PQ_bytes, use_opq,
                                            query_path, result_path, gt_path_prefix, K, Lsearch, num_bins, use_adj_sample);
        }
        else if (data_type == std::string("uint8"))
        {
            return build_and_search<uint8_t>(metric, data_path, R, Lbuild, alpha, num_threads,
                                            use_pq_build, build_PQ_bytes, use_opq,
                                            query_path, result_path, gt_path_prefix, K, Lsearch, num_bins, use_adj_sample);
        }
        else if (data_type == std::string("float"))
        {
            return build_and_search<float>(metric, data_path, R, Lbuild, alpha, num_threads,
                                            use_pq_build, build_PQ_bytes, use_opq,
                                            query_path, result_path, gt_path_prefix, K, Lsearch, num_bins, use_adj_sample);
        }
        else
        {
            std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
            return -1;
        }
    }
    catch (std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}

