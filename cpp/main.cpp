#include <iostream>

#include "marginals.h"
#include "argmax.h"

int main()
{
    diffdp::MarginalsChart chart_forward(10);
    diffdp::MarginalsAlgorithm alg(&chart_forward, false);

    const unsigned size = 4;
    float h0[] = {0.1673, 0.1490, 0.8203};
    float h1[] = {0.8221, 0.8914, 0.3122};
    float h2[] = {0.9517, 0.6141, 0.9157};

    alg.forward(size, [&] (int head, int mod) {
        if (head == 0)
        {
            if (mod == 1) return h0[0];
            if (mod == 2) return h1[1];
            if (mod == 3) return h2[2];
        }
        else
        {
            if (head == 1) return h0[mod - 1];
            if (head == 2) return h1[mod - 1];
            if (head == 3) return h2[mod - 1];
        }
        std::cout << "SHOULD NOT HAPPEN\n";
    });
    std::cout << alg.chart_forward->c_cright(0, size - 1) << "\n";
    for (int head = 1 ; head < size ; ++head)
    {
        for (int mod = 1 ; mod < size ; ++mod)
        {
            if (head == mod)
                std::cout << alg.output(0, mod) << "\t";
            else
                std::cout << alg.output(head, mod) << "\t";
        }
        std::cout << std::endl;
    }



    ArgmaxChart argmax_chart(10);
    ArgmaxAlgorithm argmax_alg(&argmax_chart, true);

    std::cout << "---\n";
    auto heads = argmax_alg.forward(size, [&] (int head, int mod) {
        if (head == 0)
        {
            if (mod == 1) return h0[0];
            if (mod == 2) return h1[1];
            if (mod == 3) return h2[2];
        }
        else
        {
            if (head == 1) return h0[mod - 1];
            if (head == 2) return h1[mod - 1];
            if (head == 3) return h2[mod - 1];
        }
        std::cout << "SHOULD NOT HAPPEN\n";
    });
    // W=2.3401
    // 2 0 2
    for (auto const h : heads)
        std::cout << h << "\n";

    return 0;
}