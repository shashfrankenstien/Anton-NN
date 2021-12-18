#pragma once

#include <vector>
#include <stdlib.h> // srand, rand and RAND_MAX
#include <stdio.h> // printf
#include <cassert> //assert
#include <cmath>


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


inline double intialize_weight()
{
    int r = rand();
    return (((double)r / (double)RAND_MAX) * 2) - 1; // random number between -1 and 1
}


// TODO: add cost function here and derivative of it

#ifndef INIT_WEIGHT_FUNC
    #define INIT_WEIGHT_FUNC intialize_weight
#endif
#ifndef ACTIVATION_FUNC
    #define ACTIVATION_FUNC Sigmoid
#endif
#ifndef ACTIVATION_DERIVATIVE_FUNC
    #define ACTIVATION_DERIVATIVE_FUNC DSigmoid
#endif

#ifndef BIAS
    #define BIAS 0.5
#endif
#ifndef LEARNING_RATE
    #define LEARNING_RATE 1
#endif

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
        void adjust_prev_weights();


    private:
        unsigned m_idx;
        Layer *m_prev_layer;
        Layer *m_next_layer;

        double m_sum_val;
        double m_activation_val;
        double m_gradient;
        std::vector<double> m_conn_weights;

        double get_activation_for(unsigned neuron_idx) const;

};


// ****************** Net ******************

class Net
{
    public:
        Net(const std::vector<unsigned> &layers);
        // ~Net();

        void feed_forward(const std::vector<double> &inp);
        void back_propagate(std::vector<double> &out);

        void show() const;

    private:
        std::vector<Layer> m_layers;
        double m_error;

};
