#include "marginals.h"

namespace diffdp
{

MarginalsChart::MarginalsChart(unsigned size) :
        size(size),
        size_3d(size*size*size),
        size_2d(size*size),
        _memory(new float[size_3d * 8 + size_2d * 8]),
        a_cleft(size, _memory),
        a_cright(size, _memory + 1u*size_3d),
        a_uleft(size, _memory + 2u*size_3d),
        a_uright(size, _memory + 3u*size_3d),
        b_cleft(size, _memory + 4u*size_3d),
        b_cright(size, _memory + 5u*size_3d),
        b_uleft(size, _memory + 6u*size_3d),
        b_uright(size, _memory + 7u*size_3d),
        c_cleft(size, _memory + 8u*size_3d),
        c_cright(size, _memory + 8u*size_3d + 1u*size_2d),
        c_uleft(size, _memory + 8u*size_3d + 2u*size_2d),
        c_uright(size, _memory + 8u*size_3d + 3u*size_2d),
        soft_c_cleft(size, _memory + 8u*size_3d + 4u*size_2d),
        soft_c_cright(size, _memory + 8u*size_3d + 5u*size_2d),
        soft_c_uleft(size, _memory + 8u*size_3d + 6u*size_2d),
        soft_c_uright(size, _memory + 8u*size_3d + 7u*size_2d)
{}

MarginalsChart::~MarginalsChart()
{
    delete[] _memory;
}

void MarginalsChart::zeros()
{
    std::fill(_memory, _memory + size_3d * 8 + size_2d * 8, float{});
}


MarginalsAlgorithm::MarginalsAlgorithm(MarginalsChart* chart_forward, const bool single_root) :
    _single_root(single_root),
    chart_forward(chart_forward)
{}


void MarginalsAlgorithm::forward_maximize(MarginalsChart* chart_forward, const unsigned size, const bool single_root)
{
    for (unsigned l = 1u; l < size; ++l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;

            if (single_root && i == 0)
                chart_forward->c_uright(i, j) += forward_entropy_reg_one_root(
                        chart_forward->c_cright.iter2(i, i), chart_forward->c_cleft.iter1(i + 1, j),
                        chart_forward->a_uright.iter3(i, j, i),
                        chart_forward->b_uright.iter3(i, j, i),
                        l
                );
            else
                chart_forward->c_uright(i, j) += forward_entropy_reg(
                        chart_forward->c_cright.iter2(i, i), chart_forward->c_cleft.iter1(i + 1, j),
                        chart_forward->a_uright.iter3(i, j, i),
                        chart_forward->b_uright.iter3(i, j, i),
                        l
                );

            if (i > 0u) // because the root cannot be a modifier
            {
                chart_forward->c_uleft(i, j) += forward_entropy_reg(
                        chart_forward->c_cright.iter2(i, i), chart_forward->c_cleft.iter1(i + 1, j),
                        chart_forward->a_uleft.iter3(i, j, i),
                        chart_forward->b_uleft.iter3(i, j, i),
                        l
                );
            }

            chart_forward->c_cright(i, j) = forward_entropy_reg(
                    chart_forward->c_uright.iter2(i, i + 1), chart_forward->c_cright.iter1(i + 1, j),
                    chart_forward->a_cright.iter3(i, j, i + 1),
                    chart_forward->b_cright.iter3(i, j, i + 1),
                    l
            );

            if (i > 0u)
            {
                chart_forward->c_cleft(i, j) = forward_entropy_reg(
                        chart_forward->c_cleft.iter2(i, i), chart_forward->c_uleft.iter1(i, j),
                        chart_forward->a_cleft.iter3(i, j, i),
                        chart_forward->b_cleft.iter3(i, j, i),
                        l
                );
            }
        }
    }
}

void MarginalsAlgorithm::forward_backtracking(MarginalsChart* chart_forward, const unsigned size, const bool single_root)
{
    chart_forward->soft_c_cright(0, size - 1) = 1.0f;

    for (unsigned l = size - 1; l >= 1; --l)
    {
        for (unsigned i = 0u; i < size - l; ++i)
        {
            unsigned j = i + l;

            diffdp::forward_backtracking(
                    chart_forward->soft_c_uright.iter2(i, i + 1), chart_forward->soft_c_cright.iter1(i + 1, j),
                    chart_forward->soft_c_cright(i, j),
                    chart_forward->b_cright.iter3(i, j, i + 1),
                    l
            );

            if (i > 0u)
            {
                diffdp::forward_backtracking(
                        chart_forward->soft_c_cleft.iter2(i, i), chart_forward->soft_c_uleft.iter1(i, j),
                        chart_forward->soft_c_cleft(i, j),
                        chart_forward->b_cleft.iter3(i, j, i),
                        l
                );
            }

            if (single_root && i == 0)
                diffdp::forward_backtracking_one_root(
                        chart_forward->soft_c_cright.iter2(i, i), chart_forward->soft_c_cleft.iter1(i + 1, j),
                        chart_forward->soft_c_uright(i, j),
                        chart_forward->b_uright.iter3(i, j, i),
                        l
                );
            else
                diffdp::forward_backtracking(
                        chart_forward->soft_c_cright.iter2(i, i), chart_forward->soft_c_cleft.iter1(i + 1, j),
                        chart_forward->soft_c_uright(i, j),
                        chart_forward->b_uright.iter3(i, j, i),
                        l
                );


            if (i > 0u)
            {
                diffdp::forward_backtracking(
                        chart_forward->soft_c_cright.iter2(i, i), chart_forward->soft_c_cleft.iter1(i + 1, j),
                        chart_forward->soft_c_uleft(i, j),
                        chart_forward->b_uleft.iter3(i, j, i),
                        l
                );
            }
        }
    }
}

float MarginalsAlgorithm::output(const unsigned head, const unsigned mod) const
{
    if (head < mod)
        return chart_forward->soft_c_uright(head, mod);
    else if (mod < head)
        return chart_forward->soft_c_uleft(mod, head);
    else
        return std::nanf("");
}


}