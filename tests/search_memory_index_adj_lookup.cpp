// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "index.h"
#include "memory_mapper.h"
#include "utils.h"

namespace po = boost::program_options;

template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &gt_ids_file, const std::string &gt_indptr_file, 
                        const uint32_t num_threads, const uint32_t recall_at, const std::vector<uint32_t> &num_pos,
                        const bool dynamic, const bool show_qps_per_thread)
{
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr, *gt_indptr = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_id_num, gt_id_dim, gt_indptr_num, gt_indptr_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    // load the ground truth file
    diskann::load_bin<uint32_t>(gt_ids_file, gt_ids, gt_id_num, gt_id_dim);
    diskann::load_bin<uint32_t>(gt_indptr_file, gt_indptr, gt_indptr_num, gt_indptr_dim);    
    assert (gt_indptr_num == query_num + 1);
    std::cout << "loaded metadata" << std::endl;

    using TagT = uint32_t;
    const bool concurrent = false, pq_dist_build = false, use_opq = false, tags = false;
    const size_t num_pq_chunks = 0;
    using IndexType = diskann::Index<T, TagT, LabelT>;
    const size_t num_frozen_pts = IndexType::get_graph_num_frozen_points(index_path);
    IndexType index(metric, query_dim, 0, dynamic, tags, concurrent, pq_dist_build, num_pq_chunks, use_opq,
                    num_frozen_pts);
    std::cout << "Index class instantiated" << std::endl;
    index.load(index_path.c_str(), num_threads, *(std::max_element(num_pos.begin(), num_pos.end())));
    std::cout << "Index loaded" << std::endl;
    if (metric == diskann::FAST_L2)
        index.optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    if (tags)
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(20) << "Mean Latency (mus)"
                  << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 20 + 15;
    }
    else
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 18 + 20 + 15;
    }
    
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(num_pos.size());
    std::vector<std::vector<float>> query_result_dists(num_pos.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint64_t> cmp_stats;

    for (uint32_t test_id = 0; test_id < num_pos.size(); test_id++)
    {
        uint32_t L = num_pos[test_id];

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res{};
        cmp_stats = std::vector<uint64_t>(query_num, 0);

        std::random_device device;
        std::mt19937       generator(device());

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            
            auto qs = std::chrono::high_resolution_clock::now();
            uint32_t start = gt_indptr[i], end = gt_indptr[i+1];
            auto len = end - start;
            std::vector<uint32_t> curr_ids(gt_ids + start, gt_ids + end);
            if (len >= L) {
                std::vector<uint32_t> samples(L);
                std::uniform_int_distribution<> dis(0, len - 1);
                for (auto& x : samples)
                    x = curr_ids[dis(generator)];
                cmp_stats[i] = (uint64_t) index.search_with_adj_lookup(query + i * query_aligned_dim,
                                                            samples, recall_at,
                                                            query_result_ids[test_id].data() + i * recall_at);
            }
            else {
                cmp_stats[i] = (uint64_t) index.search_with_adj_lookup(query + i * query_aligned_dim, 
                                                            curr_ids, recall_at,
                                                            query_result_ids[test_id].data() + i * recall_at);
            }
            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        double displayed_qps = query_num / diff.count();

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float)query_num;

        if (tags)
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float)mean_latency
                      << std::setw(15) << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        else
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
                      << std::setw(20) << (float)mean_latency << std::setw(15)
                      << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        std::cout << std::endl;
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : num_pos)
    {
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }
        std::string cur_result_path = result_path_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);
        test_id++;
    }

    diskann::aligned_free(query);

    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path, query_file, gt_ids_file, gt_indptr_file;
    uint32_t num_threads, K;
    std::vector<uint32_t> num_pos;
    bool dynamic, show_qps_per_thread;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/fast_l2/cosine>");
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                           "Path prefix to the index");
        desc.add_options()("result_path", po::value<std::string>(&result_path)->required(),
                           "Path prefix for saving results of the queries");
        desc.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                           "Query file in binary format");
        desc.add_options()("gt_ids_file", po::value<std::string>(&gt_ids_file)->required(),
                           "ground truth file for the queryset");
        desc.add_options()("gt_indptr_file", po::value<std::string>(&gt_indptr_file)->required(),
                           "ground truth file for the queryset");
        desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(), "Number of neighbors to be returned");
        desc.add_options()("search_list,L", po::value<std::vector<uint32_t>>(&num_pos)->multitoken(),
                           "List of L values of search");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads used for building index (defaults to "
                           "omp_get_num_procs())");
        desc.add_options()("dynamic", po::value<bool>(&dynamic)->default_value(false),
                           "Whether the index is dynamic. Default false.");
        desc.add_options()("qps_per_thread", po::bool_switch(&show_qps_per_thread),
                           "Print overall QPS divided by the number of threads in "
                           "the output table");

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

    diskann::Metric metric;
    if ((dist_fn == std::string("mips")) && (data_type == std::string("float")))
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
    else if ((dist_fn == std::string("fast_l2")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::FAST_L2;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                     "supported in general, and mips/fast_l2 only for floating "
                     "point data."
                  << std::endl;
        return -1;
    }

    try
    {
        if (data_type == std::string("int8"))
        {
            return search_memory_index<int8_t>(metric, index_path_prefix, result_path, query_file, 
                                                gt_ids_file, gt_indptr_file,
                                                num_threads, K, num_pos, dynamic, show_qps_per_thread);
        }
        else if (data_type == std::string("uint8"))
        {
            return search_memory_index<uint8_t>(metric, index_path_prefix, result_path, query_file, 
                                                gt_ids_file, gt_indptr_file,
                                                num_threads, K, num_pos, dynamic, show_qps_per_thread);
        }
        else if (data_type == std::string("float"))
        {
            return search_memory_index<float>(metric, index_path_prefix, result_path, query_file, 
                                                gt_ids_file, gt_indptr_file,
                                                num_threads, K, num_pos, dynamic, show_qps_per_thread);
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
