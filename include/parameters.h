// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include "omp.h"
#include "defaults.h"

namespace diskann
{

class IndexWriteParameters

{
  public:
    const uint32_t search_list_size; // L
    const uint32_t max_degree;       // R
    const bool saturate_graph;
    const uint32_t max_occlusion_size; // C
    const float alpha;
    const uint32_t num_threads;
    const uint32_t filter_list_size; // Lf
    const uint32_t num_frozen_points;
    const uint32_t max_nq_per_node;
    const float ood_lambda;
    const bool ood_build;

  private:
    IndexWriteParameters(const uint32_t search_list_size, const uint32_t max_degree, const bool saturate_graph,
                         const uint32_t max_occlusion_size, const float alpha, const uint32_t num_threads,
                         const uint32_t filter_list_size, const uint32_t num_frozen_points,
                         const uint32_t max_nq_per_node, const float ood_lambda, const bool ood_build)
        : search_list_size(search_list_size), max_degree(max_degree), saturate_graph(saturate_graph),
          max_occlusion_size(max_occlusion_size), alpha(alpha), num_threads(num_threads),
          filter_list_size(filter_list_size), num_frozen_points(num_frozen_points),
          max_nq_per_node(max_nq_per_node), ood_lambda(ood_lambda), ood_build(ood_build)
    {
    }

    friend class IndexWriteParametersBuilder;
};

class IndexWriteParametersBuilder
{
    /**
     * Fluent builder pattern to keep track of the 7 non-default properties
     * and their order. The basic ctor was getting unwieldy.
     */
  public:
    IndexWriteParametersBuilder(const uint32_t search_list_size, // L
                                const uint32_t max_degree        // R
                                )
        : _search_list_size(search_list_size), _max_degree(max_degree)
    {
    }

    IndexWriteParametersBuilder &with_max_occlusion_size(const uint32_t max_occlusion_size)
    {
        _max_occlusion_size = max_occlusion_size;
        return *this;
    }

    IndexWriteParametersBuilder &with_saturate_graph(const bool saturate_graph)
    {
        _saturate_graph = saturate_graph;
        return *this;
    }

    IndexWriteParametersBuilder &with_alpha(const float alpha)
    {
        _alpha = alpha;
        return *this;
    }

    IndexWriteParametersBuilder &with_ood_build(const bool ood_build)
    {
        _ood_build = ood_build;
        return *this;
    }

    IndexWriteParametersBuilder &with_max_nq_per_node(const uint32_t max_nq_per_node)
    {
        _max_nq_per_node = max_nq_per_node;
        return *this;
    }

    IndexWriteParametersBuilder &with_ood_lambda(const float ood_lambda)
    {
        _ood_lambda = ood_lambda;
        return *this;
    }

    IndexWriteParametersBuilder &with_num_threads(const uint32_t num_threads)
    {
        _num_threads = num_threads == 0 ? omp_get_num_threads() : num_threads;
        return *this;
    }

    IndexWriteParametersBuilder &with_filter_list_size(const uint32_t filter_list_size)
    {
        _filter_list_size = filter_list_size == 0 ? _search_list_size : filter_list_size;
        return *this;
    }

    IndexWriteParametersBuilder &with_num_frozen_points(const uint32_t num_frozen_points)
    {
        _num_frozen_points = num_frozen_points;
        return *this;
    }

    IndexWriteParameters build() const
    {
        return IndexWriteParameters(_search_list_size, _max_degree, _saturate_graph, _max_occlusion_size, _alpha,
                                    _num_threads, _filter_list_size, _num_frozen_points,
                                    _max_nq_per_node, _ood_lambda, _ood_build);
    }

    IndexWriteParametersBuilder(const IndexWriteParameters &wp)
        : _search_list_size(wp.search_list_size), _max_degree(wp.max_degree),
          _max_occlusion_size(wp.max_occlusion_size), _saturate_graph(wp.saturate_graph), _alpha(wp.alpha),
          _filter_list_size(wp.filter_list_size), _num_frozen_points(wp.num_frozen_points),
          _max_nq_per_node(wp.max_nq_per_node), _ood_lambda(wp.ood_lambda), _ood_build(wp.ood_build)
    {
    }
    IndexWriteParametersBuilder(const IndexWriteParametersBuilder &) = delete;
    IndexWriteParametersBuilder &operator=(const IndexWriteParametersBuilder &) = delete;

  private:
    uint32_t _search_list_size{};
    uint32_t _max_degree{};
    uint32_t _max_occlusion_size{defaults::MAX_OCCLUSION_SIZE};
    bool _saturate_graph{defaults::SATURATE_GRAPH};
    float _alpha{defaults::ALPHA};
    uint32_t _num_threads{defaults::NUM_THREADS};
    uint32_t _filter_list_size{defaults::FILTER_LIST_SIZE};
    uint32_t _num_frozen_points{defaults::NUM_FROZEN_POINTS_STATIC};
    uint32_t _max_nq_per_node{defaults::MAX_NQ_PER_NODE};
    float _ood_lambda{defaults::OOD_LAMBDA};
    bool _ood_build{defaults::OOD_BUILD};
};

} // namespace diskann
