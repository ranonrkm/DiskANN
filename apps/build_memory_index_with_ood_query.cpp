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
int build_in_memory_index(const diskann::Metric &metric, const std::string &data_path, 
                          const std::string &query_path, const std::string &qids_path, 
                          const uint32_t max_nq_per_node, const float ood_lambda,
                          const uint32_t R, const uint32_t L, const float alpha,
                          const std::string &save_path, const uint32_t num_threads,
                          const bool use_pq_build, const size_t num_pq_bytes, const bool use_opq)
{
    diskann::IndexWriteParameters paras = diskann::IndexWriteParametersBuilder(L, R)
                                              .with_alpha(alpha)
                                              .with_saturate_graph(false)
                                              .with_ood_build(true)
                                              .with_max_nq_per_node(max_nq_per_node)
                                              .with_ood_lambda(ood_lambda)
                                              .with_num_threads(num_threads)
                                              .build();

    size_t data_num, data_dim, query_num, query_dim, qids_num, qids_dim;
    diskann::get_bin_metadata(data_path, data_num, data_dim);
    diskann::get_bin_metadata(query_path, query_num, query_dim);
    diskann::get_bin_metadata(qids_path, qids_num, qids_dim);

    diskann::Index<T, TagT, LabelT> index(metric, data_dim, data_num, false, false, false, use_pq_build, num_pq_bytes,
                                          use_opq);
    auto s = std::chrono::high_resolution_clock::now();

    index.build_ood_index(data_path.c_str(), data_num, query_path.c_str(), query_num, qids_path.c_str(), paras);

    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    std::cout << "Indexing time: " << diff.count() << "\n";
    index.save(save_path.c_str());
    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, query_path, qids_path, index_path_prefix;
    uint32_t num_threads, R, L, build_PQ_bytes, max_nq_per_node;
    float alpha, ood_lambda;
    bool use_pq_build, use_opq;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/cosine>");
        desc.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                           "Input data file in bin format");
        desc.add_options()("query_path", po::value<std::string>(&query_path)->required(),
                           "Training query data file in bin format");
        desc.add_options()("qids_path", po::value<std::string>(&qids_path)->required(),
                           "Training query NN-ids file in bin format");
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                           "Path prefix for saving index file components");
        desc.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64), "Maximum graph degree");
        desc.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                           "Build complexity, higher value results in better graphs");
        desc.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                           "alpha controls density and diameter of graph, set "
                           "1 for sparse graph, "
                           "1.2 or 1.4 for denser graphs with lower diameter");
        desc.add_options()("max_nq", po::value<uint32_t>(&max_nq_per_node)->default_value(5), "Maximum number of queries per base point");
        desc.add_options()("lambda", po::value<float>(&ood_lambda)->default_value(0.5f),
                           "lambda controls the weight of OOD query distances ");
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

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        use_pq_build = (build_PQ_bytes > 0);
        use_opq = vm["use_opq"].as<bool>();
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

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
        diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                      << "  #threads: " << num_threads << std::endl;
        
        if (data_type == std::string("int8"))
            return build_in_memory_index<int8_t>(metric, data_path, query_path, qids_path, max_nq_per_node, ood_lambda, 
                                                    R, L, alpha, index_path_prefix, num_threads,
                                                    use_pq_build, build_PQ_bytes, use_opq);
        else if (data_type == std::string("uint8"))
            return build_in_memory_index<uint8_t>(metric, data_path, query_path, qids_path, max_nq_per_node, ood_lambda, 
                                                    R, L, alpha, index_path_prefix, num_threads,
                                                    use_pq_build, build_PQ_bytes, use_opq);
        else if (data_type == std::string("float"))
            return build_in_memory_index<float>(metric, data_path, query_path, qids_path, max_nq_per_node, ood_lambda, 
                                                    R, L, alpha, index_path_prefix, num_threads,
                                                    use_pq_build, build_PQ_bytes, use_opq);
        else
        {
            std::cout << "Unsupported type. Use one of int8, uint8 or float." << std::endl;
            return -1;
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
