#include <vector>
#include <stdio.h>
#include <stdlib.h> // srand and RAND_MAX
#include <time.h> // timestamp as seed for srand()
#include <cmath>
#include <string.h> // strncmp

#include "conf.h"
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
        // printf("%d ^ %d = %d\n", (int)A, (int)B, (int)out);
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
        // printf("%d ^ %d = %d %d\n", (int)A, (int)B, (int)out, (int)(1-out));
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
        // printf("%d ^ %d ^ %d = %d\n", (int)A, (int)B, (int)C, (int)out);
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
        // printf("%d ^ %d ^ %d = %d %d\n", (int)A, (int)B, (int)C, (int)out, (int)(1-out));
    }
}


#include "vendor/mnist/mnist_reader.hpp"
#define EMNIST_DATA_DIR "./data/EMNIST_Digits"
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


typedef enum {
    XOR,
    XOR2,
    TXOR,
    TXOR2,
    MNIST
} Test;




int main(int argc, char*argv[])
{
    if (argc <=1) {
        printf("%d argc %s", argc, argv[0]);
        return 1;
    }
    int opt_size = strnlen(argv[1], 10);
    Test opt;

    unsigned samp_size = 10000;
    int repetitions = 2;

    if (strncmp(argv[1], "xor", opt_size)==0) {
        printf("xor\n");
        opt = XOR;
    } else if (strncmp(argv[1], "xor2", opt_size)==0) {
        printf("xor2\n");
        opt = XOR2;
    } else if (strncmp(argv[1], "txor", opt_size)==0) {
        printf("txor\n");
        opt = TXOR;
    } else if (strncmp(argv[1], "txor2", opt_size)==0) {
        printf("txor2\n");
        opt = TXOR2;
    } else if (strncmp(argv[1], "mnist", opt_size)==0) {
        printf("mnist\n");
        opt = MNIST;
        samp_size = 200000;
        repetitions = 5;
    } else {
        printf("wrong\n");
        return 2;
    }

    // srand(time(NULL));
    srand(RANDOM_SEED);

    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_outputs;


    switch (opt) {
        case MNIST:
            fetch_mnist_digits(test_inputs, test_outputs, samp_size);
            break;
        case XOR:
            generate_xor_data(test_inputs, test_outputs, samp_size);
            break;
        case XOR2:
            generate_xor_double_trouble(test_inputs, test_outputs, samp_size);
            break;
        case TXOR:
            generate_ternary_xor_data(test_inputs, test_outputs, samp_size);
            break;
        case TXOR2:
            generate_ternary_xor_double_trouble(test_inputs, test_outputs, samp_size);
            break;
        default:
            return 1;
    }

    // length of 'layers' variable describes number of layers, each element describes number of neurons in the layer
    std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 50, 20, 16, 16, 16, (unsigned)test_outputs[0].size()};
    Net<Neuron> myNet(layers);

    std::vector<double> results_container;
    double abs_avg_error;

    unsigned counter = 0;
    for (unsigned r = 1; r <= repetitions; r++) {
        for (unsigned i = 0; i < test_inputs.size(); i++) {
            myNet.feed_forward(test_inputs[i]);
            myNet.back_propagate_sgd(test_outputs[i]);

            printf("[%d] inp: ", counter++);

            if (opt==MNIST) {
                printf("%d ", uint8_reconstruct(test_outputs[i]));
            } else {
                for (unsigned j = 0; j < test_inputs[i].size(); j++)
                    printf("%d ", (int)test_inputs[i][j]);
            }

            printf(" exp: ");
            for (unsigned j = 0; j < test_outputs[i].size(); j++)
                printf("%d ", (int)test_outputs[i][j]);

            printf(" -> ");
            results_container.clear();
            abs_avg_error = 0;
            myNet.get_results(results_container, abs_avg_error);

            bool overall_success = true;
            printf("RES: ");
            for (unsigned n = 0; n < results_container.size(); n++) {
                int bin_out = (results_container[n]>=0.5) ? 1 : 0;
                printf("%f(%d) ", results_container[n], bin_out);
                printf("%c ", (bin_out!=test_outputs[i][n]) ? 'x' : ' ');
                overall_success = overall_success && (bin_out==test_outputs[i][n]);
            }
            printf("ERR: %3.4f %c\n", abs_avg_error*100, (overall_success? '-': 'x'));
            // printf("**************\n");
        }
    }

    return 0;
}
