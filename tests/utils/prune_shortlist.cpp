// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include "tsl/robin_set.h"
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

template<typename T>
void occlude_shortlist(
    diskann::Distance<T> *distance_fn, diskann::Metric& dist_metric,
    const T* base, const size_t aligned_dim,
    std::vector<diskann::Neighbor> &pool, const float alpha,
    const unsigned degree, std::vector<unsigned> &result) {
  if (pool.size() == 0)
    return;

  // Truncate pool at maxc and initialize scratch spaces
  // assert(std::is_sorted(pool.begin(), pool.end()));
  assert(result.size() == 0);
  std::vector<float> occlude_factor(pool.size());
  occlude_factor.clear();

  float cur_alpha = 1;
  while (cur_alpha <= alpha && result.size() < degree) {
    // used for MIPS, where we store a value of eps in cur_alpha to
    // denote pruned out entries which we can skip in later rounds.
    float eps = cur_alpha + 0.01f;

    for (auto iter = pool.begin();
          result.size() < degree && iter != pool.end(); ++iter) {
      if (occlude_factor[iter - pool.begin()] > cur_alpha) {
        continue;
      }
      // Set the entry to float::max so that is not considered again
      occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
      result.push_back(iter - pool.begin());

      // Update occlude factor for points from iter+1 to pool.end()
      for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
        auto t = iter2 - pool.begin();
        if (occlude_factor[t] > alpha)
          continue;
        float djk =
            distance_fn->compare(base + aligned_dim * (size_t) iter2->id,
                                base + aligned_dim * (size_t) iter->id,
                                (unsigned) aligned_dim);
        if (dist_metric == diskann::Metric::L2 ||
            dist_metric == diskann::Metric::COSINE) {
          occlude_factor[t] =
              (djk == 0) ? std::numeric_limits<float>::max()
                          : std::max(occlude_factor[t], iter2->distance / djk);
        } else if (dist_metric == diskann::Metric::INNER_PRODUCT) {
          // Improvization for flipping max and min dist for MIPS
          float x = -iter2->distance;
          float y = -djk;
          if (y > cur_alpha * x) {
            occlude_factor[t] = std::max(occlude_factor[t], eps);
          }
        }
      }
    }
    cur_alpha *= 1.4;
  }
}

template<typename T>
void prune_shortlist(diskann::Distance<T> *distance_fn,
                     diskann::Metric& metric,
                     const T* query, const T* base,
                     const size_t aligned_dim,
                     const unsigned* nnids, const size_t L,
                     unsigned* final_ids, float* final_dists,
                     const size_t K, const float alpha) {
  std::vector<diskann::Neighbor> pool(L); 
  for (unsigned i = 0; i < (unsigned) L; i++) {
    unsigned id = nnids[i];
    float distance = distance_fn->compare(base + aligned_dim * (size_t) id,
                                          query, (unsigned) aligned_dim);
    pool[i] = diskann::Neighbor(id, distance);
  }
  std::vector<unsigned> result;
  occlude_shortlist(distance_fn, metric, base, aligned_dim, 
                    pool, alpha, K, result);
  size_t pos = 0;
  tsl::robin_set<unsigned> inserted;
  for (auto& i : result) {
    diskann::Neighbor p = pool[i];
    final_ids[pos] = p.id;
    final_dists[pos++] = p.distance;
    inserted.insert(i);
  }
  for (unsigned i = 0; pos < K && i < (unsigned)L; i++) {
    if (inserted.find(i) == inserted.end()) {
      diskann::Neighbor p = pool[i];
      final_ids[pos] = p.id;
      final_dists[pos++] = p.distance;
    }
  }
}

template<typename T>
void aux_main(const std::string &base_file, const std::string &query_file,
             const std::string &nnids_file, size_t k,
             diskann::Metric &metric, const float alpha,
             const std::string& result_path_prefix,
             size_t num_threads) {
  // Load the query file
  T*        query = nullptr;
  size_t    query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  // Load the base file
  T*        base = nullptr;
  size_t    base_num, base_dim, base_aligned_dim;
  diskann::load_aligned_bin<T>(base_file, base, base_num, base_dim,
                               base_aligned_dim);
  assert(query_dim == base_dim);

  // Load the nnids file
  unsigned* nnids = nullptr;
  size_t nnids_num, nnids_dim;
  diskann::load_bin<unsigned>(nnids_file, nnids, nnids_num, nnids_dim);

  diskann::Distance<T> *distance_fn = diskann::get_distance_function<T>(metric);  
  std::vector<unsigned>  query_result_ids(k * query_num);
  std::vector<float>  query_result_dists(k * query_num);
  auto s = std::chrono::high_resolution_clock::now();
  omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    prune_shortlist(distance_fn, metric,
                    query + i * query_aligned_dim,
                    base,
                    base_aligned_dim,
                    nnids + i * nnids_dim, nnids_dim, 
                    query_result_ids.data() + i * k,
                    query_result_dists.data() + i * k,
                    k, alpha);
  }
  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;
  std::cout << "Done pruning in " << (float) (diff.count() * 1000000) << " secs. Now saving results " << std::endl;

  std::string result_path =
      result_path_prefix + "_" + std::to_string(nnids_dim) + "_idx_uint32.bin";
  diskann::save_bin<_u32>(result_path, query_result_ids.data(),
                          query_num, k);
  result_path =
      result_path_prefix + "_" + std::to_string(nnids_dim) + "_scr_float.bin";
  diskann::save_bin<float>(result_path, query_result_dists.data(),
                          query_num, k);
}

int main(int argc, char **argv) {
  std::string data_type, dist_fn, base_file, query_file, shortlist_file, result_path_prefix;
  uint64_t    K;
  uint32_t    num_threads;
  float       alpha;

  try {
    po::options_description desc{"Arguments"};

    desc.add_options()("help,h", "Print information on arguments");

    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips>");
    desc.add_options()("base_file",
                       po::value<std::string>(&base_file)->required(),
                       "File containing the base vectors in binary format");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "File containing the query vectors in binary format");
    desc.add_options()(
        "shortlist_file", po::value<std::string>(&shortlist_file)->required(),
        "File name of shortlist items in binary format");
    desc.add_options()("K", po::value<uint64_t>(&K)->required(),
                       "Number of ground truth nearest diskann::neighbors to compute");
    desc.add_options()(
        "result_path_prefix", po::value<std::string>(&result_path_prefix)->required(),
        "File name for the writing final shortlisted items in binary format");
    desc.add_options()(
        "alpha", po::value<float>(&alpha)->default_value(1.2f),
        "alpha controls density and diameter of graph, set 1 for sparse graph, "
        "1.2 or 1.4 for denser graphs with lower diameter");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("mips")) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cerr << "Unsupported distance function. Use l2/mips/cosine."
              << std::endl;
    return -1;
  }

  try {
    if (data_type == std::string("float"))
      aux_main<float>(base_file, query_file, shortlist_file, K, metric, alpha, result_path_prefix, num_threads);
    if (data_type == std::string("int8"))
      aux_main<int8_t>(base_file, query_file, shortlist_file, K, metric, alpha, result_path_prefix, num_threads);
    if (data_type == std::string("uint8"))
      aux_main<uint8_t>(base_file, query_file, shortlist_file, K, metric, alpha, result_path_prefix, num_threads);
  } catch (const std::exception &e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Pruning shortlisted items failed." << std::endl;
    return -1;
  }
}
