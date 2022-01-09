#include <deque>
#include "conf.h"
#include "nn.h"
#include "test_helper.h"


void recurrent_bit_series(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples)
{
    std::deque<double> d = {1,0,1,0,0,0,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,1};
    for (int i = 0; i < n_samples; i++) {

        inputs.push_back({d[0], d[1], d[2], d[3]});
        outputs.push_back({d[4], d[5], d[6], d[7]});

        // rotate
        double val = d.back();
        d.pop_back();
        d.push_front(val);
    }
}




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

// does some flipping
void frame_from_mnist(std::vector<double> &mnist_img, ConvFrame &convf)
{
    unsigned c = 0;
    for (unsigned i = 0; i < mnist_img.size(); i++){
        convf(i%28, c) = mnist_img[i] / 255.0;
        if ((i+1)%28==0) c++;
    }
}


void fetch_mnist_digits_conv(std::vector<std::vector<ConvFrame>> &inputs, std::vector<std::vector<double>> &outputs, unsigned n_samples)
{
    auto dataset = mnist::read_dataset<std::vector, std::vector, double, uint8_t>(EMNIST_DATA_DIR);

    n_samples = MIN(dataset.training_labels.size(), n_samples);
    for (unsigned idx = 0; idx < n_samples; idx++)
    {

        ConvFrame out(28, 28);
        frame_from_mnist(dataset.training_images[idx], out);
        inputs.push_back(std::vector<ConvFrame>{out});

        std::vector<double> binrep;
        uint8_breakdown(dataset.training_labels[idx], binrep);
        outputs.push_back(binrep);
    }
}





int main()
{
    unsigned samp_size = 30000;
    unsigned repetitions = 1;

    srand(RANDOM_SEED);

    std::vector<std::vector<ConvFrame>> test_inputs;
    std::vector<std::vector<double>> test_outputs;

    printf("loading mnist..\n");
    fetch_mnist_digits_conv(test_inputs, test_outputs, 20);
    printf("loaded mnist..\n");

    std::vector<ConvTopology> topo;

    topo.push_back({.n_neurons=1, .kernel_size=3}); // input neuron
    topo.push_back({.n_neurons=4, .kernel_size=5, .stride=2});
    topo.push_back({.n_neurons=7, .kernel_size=7, .stride=1});
    // topo.push_back({.n_neurons=3, .kernel_size=7, .stride=2});
    topo.push_back({.n_neurons=4, .kernel_size=0, .pooling=ConvTopology::MAX_POOL}); // output neuron

    // std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 50, 20, 16, 10, (unsigned)test_outputs[0].size()};
    Net<ConvNeuron, ConvFrame> anton_nn(topo, {28,28});

    unsigned test = 0;

    anton_nn.feed_forward(test_inputs[test]);

    printf(" exp: ");
    for (unsigned j = 0; j < test_outputs[test].size(); j++)
        printf("%d ", (int)test_outputs[test][j]);


    std::vector<double> results_container;
    double abs_avg_error;

    anton_nn.get_results(results_container, abs_avg_error);

    bool overall_success = true;
    printf("RES: ");
    for (unsigned n = 0; n < results_container.size(); n++) {
        int bin_out = (results_container[n]>=0.5) ? 1 : 0;
        printf("%f(%d) ", results_container[n], bin_out);
        printf("%c ", (bin_out!=test_outputs[test][n]) ? 'x' : ' ');
        overall_success = overall_success && (bin_out==test_outputs[test][n]);
    }
    printf("ERR: %3.4f %c\n", abs_avg_error*100, (overall_success? '-': 'x'));

    // train_net<RecurrentNeuron>(anton_nn, test_inputs, test_outputs, repetitions, [&test_inputs](unsigned idx){input_printer(test_inputs[idx]);});

}
