#pragma once

#include "chart.h"
#include <stdexcept>
struct ArgmaxChart
{
    const unsigned size;
    const unsigned size_2d;
    float* _float_memory;
    unsigned* _unsigned_memory;
    diffdp::Matrix<float> w_cleft, w_cright, w_uleft, w_uright;
    diffdp::Matrix<unsigned> b_cleft, b_cright, b_uleft, b_uright;

    ArgmaxChart(unsigned size);
    ~ArgmaxChart();

    void zeros();
};

struct ArgmaxAlgorithm
{
    const bool _single_root;
    ArgmaxChart* chart_forward;

    ArgmaxAlgorithm(ArgmaxChart* chart_forward, const bool single_root);

    // this is kept as a separate function because
    // I think that templatized constructor are problematic
    template<class Functor>
    std::vector<unsigned> forward(const unsigned size, Functor&& weight_callback);

    static void forward_maximize(ArgmaxChart* chart_forward, const unsigned size, const bool single_root);
    static std::vector<unsigned> forward_backtracking(ArgmaxChart* chart_forward, const unsigned size);
};



// templates implementations
template<class Functor>
std::vector<unsigned> ArgmaxAlgorithm::forward(const unsigned size, Functor&& weight_callback)
{
    if (size > chart_forward->size)
        throw std::runtime_error("Chart is too small");

    // this initialization seems ok, but check why it works!
    chart_forward->zeros(); // we could skip some zeros here
    for (unsigned i = 0; i < size; ++i)
    {
        for (unsigned j = 1; j < size; ++j)
        {
            if (i < j)
                chart_forward->w_uright(i, j) = weight_callback(i, j);
            else if (j < i)
                chart_forward->w_uleft(j, i) = weight_callback(i, j);
        }
    }

    ArgmaxAlgorithm::forward_maximize(chart_forward, size, _single_root);
    return ArgmaxAlgorithm::forward_backtracking(chart_forward, size);
}

