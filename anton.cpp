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
void generate_xor_data(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples)
{
    bool A, B = 0;
    for (int i = 0; i < n_samples; i++) {
        A = (i % 2 == 0) ? 1 - A : A;
        B = i % 2;

        inputs.push_back({(double)A, (double)B});

        double out = (double)Xor(A,B);
        outputs.push_back({out});
        printf("%d ^ %d = %d\n", (int)A, (int)B, (int)out);
    }
}

void generate_xor_double_trouble(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples)
{
    bool A, B = 0;
    for (int i = 0; i < n_samples; i++) {
        A = (i % 2 == 0) ? 1 - A : A;
        B = i % 2;

        inputs.push_back({(double)A, (double)B});

        double out = (double)Xor(A,B);
        outputs.push_back({out, 1-out});
        printf("%d ^ %d = %d %d\n", (int)A, (int)B, (int)out, (int)(1-out));
    }
}


int TernaryXor(int a, int b, int c)
{
    return !(a&&b&&c) && (a^b^c);
}
void generate_ternary_xor_data(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples)
{
    bool A, B, C = 0;
    for (int i = 0; i < n_samples; i++) {
        A = (i % 4 == 0) ? 1 - A : A;
        B = (i % 2 == 0) ? 1 - B : B;
        C = i % 2;

        inputs.push_back({(double)A, (double)B, (double)C});

        double out = (double)TernaryXor(A,B,C);
        outputs.push_back({out});
        printf("%d ^ %d ^ %d = %d\n", (int)A, (int)B, (int)C, (int)out);
    }
}
void generate_ternary_xor_double_trouble(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples)
{
    bool A, B, C = 0;
    for (int i = 0; i < n_samples; i++) {
        A = (i % 4 == 0) ? 1 - A : A;
        B = (i % 2 == 0) ? 1 - B : B;
        C = i % 2;

        inputs.push_back({(double)A, (double)B, (double)C});

        double out = (double)TernaryXor(A,B,C);
        outputs.push_back({out, 1-out});
        printf("%d ^ %d ^ %d = %d %d\n", (int)A, (int)B, (int)C, (int)out, (int)(1-out));
    }
}


#include "vendor/mnist/mnist_reader.hpp"
#define EMNIST_DATA_DIR "/home/shashankgopikrishna/projects/Anton/data/EMNIST_Digits/New"
void uint8_breakdown(uint8_t val, std::vector<double> &res)
{
    for (int i = 0; i < 4; i++) {
        res.push_back((val & 0x01 == 0x01) ? 1 : 0);
        val >>= 1;
    }
}
uint8_t uint8_reconstruct(std::vector<double> &res)
{
    uint8_t val = 0x00;
    for (int i = 3; i >= 0; i--) {
        // res.push_back((val & 0x01 == 0x01) ? 1 : 0);
        val |= (((res[i]>0) ? 0x01 : 0x00));
        if(i>0)
            val <<= 1;
    }
    return val;
}
void fetch_mnist_digits(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples)
{
    auto dataset = mnist::read_dataset<std::vector, std::vector, double, uint8_t>(EMNIST_DATA_DIR);

    n_samples = MIN(dataset.training_labels.size(), n_samples);
    for (unsigned idx = 0; idx < n_samples; idx++)
    {
        std::vector<double> img;
        for (int i=0; i < 28*28; i++)
            img.push_back(dataset.training_images[idx][i] / 255.0);
        inputs.push_back(img);

        std::vector<double> binrep;
        uint8_breakdown(dataset.training_labels[idx], binrep);
        outputs.push_back(binrep);
    }
}







int main()
{
    srand(time(NULL));

    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_outputs;

    fetch_mnist_digits(test_inputs, test_outputs, 240000);

    // length of 'layers' variable describes number of layers, each element describes number of neurons in the layer
    std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 50, 20, 16, 16, (unsigned)test_outputs[0].size()};
    Net myNet(layers);

    int repetitions = 5;

    for (int r = 1; r <= repetitions; r++) {
        for (unsigned i = 0; i < test_inputs.size(); i++) {
            myNet.feed_forward(test_inputs[i]); // feed forward??

            myNet.back_propagate(test_outputs[i]);

            printf("[%d] inp: ", i*r);

            printf("%d ", uint8_reconstruct(test_outputs[i]));
            // for (unsigned j = 0; j < test_inputs[i].size(); j++)
            //     printf("%d ", (int)test_inputs[i][j]);

            printf(" exp: ");
            for (unsigned j = 0; j < test_outputs[i].size(); j++)
                printf("%d ", (int)test_outputs[i][j]);

            printf(" -> ");
            myNet.show(test_outputs[i]);
            // printf("**************\n");
        }
    }

    return 0;
}


// (time make run > ff.txt) |& awk '{print "Anton finished in", $5, "seconds"}' | tee >(spd-say -e) | while read OUTPUT; do notify-send "$OUTPUT"; done
