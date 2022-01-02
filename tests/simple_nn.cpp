#include <string.h> // strncmp

#include "conf.h"
#include "nn.h"
#include "test_helper.h"

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


void input_printer(std::vector<double> &arr)
{
    for (unsigned j = 0; j < arr.size(); j++)
        printf("%d", (int)arr[j]);
}


int main(int argc, char*argv[])
{

    unsigned samp_size = 30000;
    unsigned repetitions = 1;

    srand(RANDOM_SEED);

    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_outputs;

    if (argc <=1 || strncmp(argv[1], "xor", strnlen(argv[1], 10))==0) {
        printf("xor\n");
        generate_xor_data(test_inputs, test_outputs, samp_size);
    }
    else if (strncmp(argv[1], "xor2", strnlen(argv[1], 10))==0) {
        printf("xor2\n");
        generate_xor_double_trouble(test_inputs, test_outputs, samp_size);
    }
    else if (strncmp(argv[1], "txor", strnlen(argv[1], 10))==0) {
        printf("txor\n");
        generate_ternary_xor_data(test_inputs, test_outputs, samp_size);
    }
    else if (strncmp(argv[1], "txor2", strnlen(argv[1], 10))==0) {
        printf("txor2\n");
        generate_ternary_xor_double_trouble(test_inputs, test_outputs, samp_size);
    }
    else {
        return 1;
    }

    std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 50, 20, 16, 10, (unsigned)test_outputs[0].size()};
    Net<Neuron> myNet(layers);
    run_test<Neuron>(myNet, test_inputs, test_outputs, repetitions, [&test_inputs](unsigned idx){input_printer(test_inputs[idx]);});
}
