#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <exception>

#include "chart.h"
#include "deduction_operations.h"

namespace diffdp
{

struct MarginalsChart
{
    const unsigned size;
    const unsigned size_3d;
    const unsigned size_2d;
    float* _memory = nullptr;

    Tensor3D<float>
            a_cleft, a_cright, a_uleft, a_uright,
            b_cleft, b_cright, b_uleft, b_uright;

    Matrix<float>
            c_cleft, c_cright, c_uleft, c_uright,
            soft_c_cleft, soft_c_cright, soft_c_uleft, soft_c_uright
    ;

    MarginalsChart(unsigned size);
    ~MarginalsChart();

    void zeros();
};


struct MarginalsAlgorithm
{
    const bool _single_root;
    MarginalsChart* chart_forward;

    MarginalsAlgorithm(MarginalsChart* chart_forward, const bool single_root);

    // this is kept as a separate function because
    // I think that templatized constructor are problematic
    template<class Functor>
    double forward(const unsigned size, Functor&& weight_callback);

    static void forward_maximize(MarginalsChart* chart_forward, const unsigned size, const bool single_root);
    static void forward_backtracking(MarginalsChart* chart_forward, const unsigned size, const bool single_root);

    float output(const unsigned head, const unsigned mod) const;

};


// templates implementations
template<class Functor>
double MarginalsAlgorithm::forward(const unsigned size, Functor&& weight_callback)
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
                chart_forward->c_uright(i, j) = weight_callback(i, j);
            else if (j < i)
                chart_forward->c_uleft(j, i) = weight_callback(i, j);
        }
    }

    MarginalsAlgorithm::forward_maximize(chart_forward, size, _single_root);
    auto log_z = chart_forward->c_cright(0, size - 1);
    MarginalsAlgorithm::forward_backtracking(chart_forward, size, _single_root);
    return log_z;
}


}

