#include <vector>
#include <stdio.h>
#include <stdlib.h> // srand and RAND_MAX
#include <time.h> // timestamp as seed for srand()
#include <cmath>


#include "nn.h"



int Xor(int a, int b)
{
    return !a != !b;
}
void generate_xor_data(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs)
{
    bool A, B = 0;
    for (int i = 0; i < 10000; i++) {
        A = (i % 2 == 0) ? 1 - A : A;
        B = i % 2;

        inputs.push_back({(double)A, (double)B});

        double out = (double)Xor(A,B);
        outputs.push_back({out});
        printf("%d ^ %d = %d\n", (int)A, (int)B, Xor(A,B));
    }
}


int TernaryXor(int a, int b, int c)
{
    return !(a&&b&&c) && (a^b^c);
}
void generate_ternary_xor_data(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs)
{
    bool A, B, C = 0;
    for (int i = 0; i < 10000; i++) {
        A = (i % 4 == 0) ? 1 - A : A;
        B = (i % 2 == 0) ? 1 - B : B;
        C = i % 2;

        inputs.push_back({(double)A, (double)B, (double)C});

        double out = (double)TernaryXor(A,B,C);
        outputs.push_back({out});
        printf("%d ^ %d ^ %d = %d\n", (int)A, (int)B, (int)C, TernaryXor(A,B,C));
    }
}



int main()
{
    srand(time(NULL));

    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_outputs;

    generate_ternary_xor_data(test_inputs, test_outputs);

    // length of 'layers' variable describes number of layers, each element describes number of neurons in the layer
    std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 6, 6, (unsigned)test_outputs[0].size()};
    Net myNet(layers);

    for (unsigned i = 0; i < test_inputs.size(); i++) {
        myNet.feed_forward(test_inputs[i]); // feed forward??

        myNet.back_propagate(test_outputs[i]);

        printf("**************\n");
        printf("[%d] inputs: ", i);
        for (unsigned j = 0; j < test_inputs[i].size(); j++)
            printf("%d ", (int)test_inputs[i][j]);

        printf(", expected outputs: ");
        for (unsigned j = 0; j < test_outputs[i].size(); j++)
            printf("%d ", (int)test_outputs[i][j]);

        printf(" -> ");
        myNet.show();
        printf("**************\n");
    }
    // std::vector<double> inputs{1,1};
    // myNet.feed_forward(inputs); // feed forward??

    // std::vector<double> outputs{0};
    // myNet.back_propagate(outputs);

    // myNet.show();

    return 0;
}

