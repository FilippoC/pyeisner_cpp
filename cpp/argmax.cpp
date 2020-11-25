#include "argmax.h"

#include <limits>
#include <iostream>

ArgmaxChart::ArgmaxChart(unsigned size) :
        size(size),
        size_2d(size*size),
        _float_memory(new float[size_2d * 4]),
        _unsigned_memory(new unsigned[size_2d * 4]),
        w_cleft(size, _float_memory),
        w_cright(size, _float_memory + 1u * size_2d),
        w_uleft(size, _float_memory + 2u * size_2d),
        w_uright(size, _float_memory + 3u * size_2d),
        b_cleft(size, _unsigned_memory),
        b_cright(size, _unsigned_memory + 1u * size_2d),
        b_uleft(size, _unsigned_memory + 2u * size_2d),
        b_uright(size, _unsigned_memory + 3u * size_2d)
{}


ArgmaxChart::~ArgmaxChart()
{
    delete[] _float_memory;
    delete[] _unsigned_memory;
}


void ArgmaxChart::zeros()
{
    std::fill(_float_memory, _float_memory + size_2d * 4, float{});
    std::fill(_unsigned_memory, _unsigned_memory + size_2d * 4, unsigned{});
}

ArgmaxAlgorithm::ArgmaxAlgorithm(ArgmaxChart* chart_forward, const bool single_root) :
        _single_root(single_root),
        chart_forward(chart_forward)
{}



void ArgmaxAlgorithm::forward_maximize(ArgmaxChart* chart_forward, const unsigned size, const bool single_root)
{
    for (unsigned l = 1u; l < size; ++l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;

            if (single_root && i == 0)
            {
                chart_forward->w_uright(i, j) += chart_forward->w_cright(i, i) + chart_forward->w_cleft(i + 1, j);
                chart_forward->b_uright(i, j) = i;
            }
            else
            {
                float best_w = -std::numeric_limits<float>::infinity();
                unsigned best_k = 0;
                for (unsigned k = i ; k < j ; ++k)
                {
                    float w = chart_forward->w_cright(i, k) + chart_forward->w_cleft(k + 1, j);
                    if (w > best_w)
                    {
                        best_w = w;
                        best_k = k;
                    }
                }
                chart_forward->w_uright(i, j) += best_w;
                chart_forward->b_uright(i, j) = best_k;
            }

            if (i > 0u) // because the root cannot be a modifier
            {
                float best_w = -std::numeric_limits<float>::infinity();
                unsigned best_k = 0;

                for (unsigned k = i ; k < j ; ++k)
                {
                    float w = chart_forward->w_cright(i, k) + chart_forward->w_cleft(k + 1, j);
                    if (w > best_w)
                    {
                        best_w = w;
                        best_k = k;
                    }
                }
                chart_forward->w_uleft(i, j) += best_w;
                chart_forward->b_uleft(i, j) = best_k;
            }

            {
                float best_w = -std::numeric_limits<float>::infinity();
                unsigned best_k = 0;
                for (unsigned k = i ; k < j ; ++k)
                {
                    float w = chart_forward->w_uright(i, k + 1) + chart_forward->w_cright(k + 1, j);
                    if (w > best_w)
                    {
                        best_w = w;
                        best_k = k;
                    }
                }
                chart_forward->w_cright(i, j) = best_w;
                chart_forward->b_cright(i, j) = best_k;
            }

            if (i > 0u)
            {
                float best_w = -std::numeric_limits<float>::infinity();
                unsigned best_k = 0;
                for (unsigned k = i ; k < j ; ++k)
                {
                    float w = chart_forward->w_cleft(i, k) + chart_forward->w_uleft(k, j);
                    if (w > best_w)
                    {
                        best_w = w;
                        best_k = k;
                    }
                }
                chart_forward->w_cleft(i, j) = best_w;
                chart_forward->b_cleft(i, j) = best_k;
            }
        }
    }
}

namespace
{
void backtrack_cright(std::vector<unsigned>& arcs, ArgmaxChart* chart_forward, int i, int j);
void backtrack_cleft(std::vector<unsigned>& arcs, ArgmaxChart* chart_forward, int i, int j);
void backtrack_uright(std::vector<unsigned>& arcs, ArgmaxChart* chart_forward, int i, int j);
void backtrack_uleft(std::vector<unsigned>& arcs, ArgmaxChart* chart_forward, int i, int j);

void backtrack_cright(std::vector<unsigned>& arcs, ArgmaxChart* chart_forward, int i, int j)
{
    if (i == j)
        return;

    auto k = chart_forward->b_cright(i, j);

    backtrack_uright(arcs, chart_forward, i, k + 1);
    backtrack_cright(arcs, chart_forward, k + 1, j);
}

void backtrack_cleft(std::vector<unsigned>& arcs, ArgmaxChart* chart_forward, int i, int j)
{
    if (i == j)
        return;

    auto k = chart_forward->b_cleft(i, j);

    backtrack_cleft(arcs, chart_forward, i, k);
    backtrack_uleft(arcs, chart_forward, k, j);
}

void backtrack_uright(std::vector<unsigned>& arcs, ArgmaxChart* chart_forward, int i, int j)
{
    if (i == 0)
        arcs.at(j - 1) = j - 1;
    else
        arcs.at(j - 1) = i - 1;

    auto k = chart_forward->b_uright(i, j);
    backtrack_cright(arcs, chart_forward, i, k);
    backtrack_cleft(arcs, chart_forward, k + 1, j);
}

void backtrack_uleft(std::vector<unsigned>& arcs, ArgmaxChart* chart_forward, int i, int j)
{
    arcs.at(i - 1) = j - 1;

    auto k = chart_forward->b_uleft(i, j);
    backtrack_cright(arcs, chart_forward, i, k);
    backtrack_cleft(arcs, chart_forward, k + 1, j);
}


}

std::vector<unsigned> ArgmaxAlgorithm::forward_backtracking(ArgmaxChart* chart_forward, const unsigned size)
{
    std::vector<unsigned> ret(size - 1);
    backtrack_cright(ret, chart_forward, 0, size - 1);
    return ret;
}
