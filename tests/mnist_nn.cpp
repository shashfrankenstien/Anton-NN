#include "conf.h"
#include "nn.h"
#include "test_helper.h"


#include "../vendor/mnist/mnist_reader.hpp"
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





void input_printer(std::vector<double> &arr)
{
    printf("%d ", uint8_reconstruct(arr));
}


int main()
{
    unsigned samp_size = 240000;
    unsigned repetitions = 1;

    srand(RANDOM_SEED);

    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_outputs;
    fetch_mnist_digits(test_inputs, test_outputs, samp_size);

    std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 160, 80, 20, (unsigned)test_outputs[0].size()};
    // std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 50, 20, 16, 10, (unsigned)test_outputs[0].size()};
    Net<Neuron> myNet(layers);
    train_net<Neuron>(myNet, test_inputs, test_outputs, repetitions, [&test_outputs](unsigned idx){input_printer(test_outputs[idx]);});
}
