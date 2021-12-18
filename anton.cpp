#include <vector>
#include <stdio.h>
#include <stdlib.h> // srand and RAND_MAX
#include <time.h> // timestamp as seed for srand()
#include <cmath>



#include "nn.h"


void generate_training_data(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs)
{
    double A, B;
    for (int i = 0; i < 10000; i++) {
        A = i % 2;
        B = (i % 3 == 0) ? 1 - B : B;

        inputs.push_back({A, B});

        double out = (double)!A != !B;
        outputs.push_back({out});
        printf("%d ^ %d = %d\n", (int)A, (int)B, !A != !B);
    }
}


int main()
{
    srand(time(NULL));

    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_outputs;

    generate_training_data(test_inputs, test_outputs);

    // length of 'layers' variable describes number of layers, each element describes number of neurons in the layer
    std::vector<unsigned> layers{2, 6, 1};
    Net myNet(layers);

    for (unsigned i = 0; i < test_inputs.size(); i++) {
        myNet.feed_forward(test_inputs[i]); // feed forward??

        myNet.back_propagate(test_outputs[i]);

        printf("**************\n");
        printf("[%d] inputs: %d %d, expected output: %d  -> ", i, (int)test_inputs[i][0], (int)test_inputs[i][1], (int)test_outputs[i][0]);
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

