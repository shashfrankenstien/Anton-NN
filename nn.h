#pragma once

#include <vector>
#include <stdlib.h> // srand, rand and RAND_MAX
#include <stdio.h> // printf
#include <string>
#include <cassert> //assert
#include <cmath>

#define PRINT_DEBUG_MSGS false

#define MIN(a, b) ((a<b)?a:b)
#define MAX(a, b) ((a>b)?a:b)


class Neuron;
typedef std::vector<Neuron> Layer;


// ****************** Defaults ******************

inline double Sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}
inline double DSigmoid(double z)
{
    return z * (1-z);
}

inline double ReLU(double z)
{
    return MAX(0, z);
}
inline double DReLU(double z)
{
    return (z>0) ? 1 : 0;
}


inline double TanH(double z)
{
    return tanh(z);
}
inline double DTanH(double z)
{
    return 1 - (z*z);
}


inline double random_weights()
{
    int r = rand();
    return (((double)r / (double)RAND_MAX) * 2) - 1; // random number between -1 and 1
}

inline double large_random_weights()
{
    int r = rand();
    return (((double)r / (double)RAND_MAX) * 8) - 4; // random number between -4 and 4
}


// TODO: add cost function option and derivative of it here

#define INIT_WEIGHT_FUNC large_random_weights
#define ACTIVATION_FUNC Sigmoid
#define ACTIVATION_DERIVATIVE_FUNC DSigmoid

#define RANDOM_SEED 0
#define BIAS 0.5
#define LEARNING_RATE 0.1
#define MOMENTUM_ALPHA 0


// ****************** Neuron ******************

class Neuron
{
    public:
        Neuron(unsigned idx, Layer *prev_layer, Layer *next_layer, unsigned next_layer_size);

        void set_value(double val);
        double get_value() const;

        void activate();
        void calc_output_gradient(double sum_errors);
        void calc_hidden_gradient();
        void adjust_input_weights();


    private:
        unsigned m_idx;
        Layer *m_prev_layer;
        Layer *m_next_layer;

        double m_sum_val;
        double m_activation_val;
        double m_gradient;
        std::vector<double> m_conn_weights;
        std::vector<double> m_old_conn_weight_deltas;

};


// ****************** Net ******************

class Net
{
    public:
        Net(const std::vector<unsigned> &layers);
        // ~Net();

        void feed_forward(const std::vector<double> &inp);
        void back_propagate_sgd(std::vector<double> &out);

        void get_results(std::vector<double> &results, double &avg_abs_error) const;

        void to_file(const std::string &filepath) const;

    private:
        std::vector<Layer> m_layers;
        double m_error;
        double m_avg_abs_error;

};
