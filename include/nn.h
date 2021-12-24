#pragma once

#include <vector>
#include <stdio.h> // printf
#include <string>
#include <cassert> //assert
#include <cmath>

#define PRINT_DEBUG_MSGS false

#define debug_print(...) \
            do { if (PRINT_DEBUG_MSGS) printf( __VA_ARGS__); } while (0)


#define MIN(a, b) ((a<b)?a:b)
#define MAX(a, b) ((a>b)?a:b)


class Neuron;
typedef std::vector<Neuron> Layer;



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
