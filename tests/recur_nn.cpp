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


void input_printer(std::vector<double> &arr)
{
    for (unsigned j = 0; j < arr.size(); j++)
        printf("%d ", (int)arr[j]);
}


int main()
{
    unsigned samp_size = 30000;
    unsigned repetitions = 1;

    srand(RANDOM_SEED);

    std::vector<std::vector<double>> test_inputs;
    std::vector<std::vector<double>> test_outputs;
    recurrent_bit_series(test_inputs, test_outputs, samp_size);

    std::vector<unsigned> layers{(unsigned)test_inputs[0].size(), 50, 20, 16, 10, (unsigned)test_outputs[0].size()};
    Net<RecurrentNeuron> myNet(layers);
    run_test<RecurrentNeuron>(myNet, test_inputs, test_outputs, repetitions, [&test_inputs](unsigned idx){input_printer(test_inputs[idx]);});
}
