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
        void calc_output_gradient(double target);
        void calc_hidden_gradient();
        void adjust_input_weights();


    protected:
        unsigned m_idx;
        Layer *m_prev_layer;
        Layer *m_next_layer;

        double m_activation_val;
        double m_gradient;
        std::vector<double> m_conn_weights;
        std::vector<double> m_old_conn_weight_deltas;

        double get_activation_for(unsigned other_idx) const;
        void adjust_weight_for(unsigned other_idx, double other_gradient);

};





// ****************** Net ******************

template <class N>
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



template <class N>
Net<N>::Net(const std::vector<unsigned> &layers)
: m_error(0)
{
    unsigned num_layers = layers.size();

    // create Layers in each layer
    for (unsigned layer_idx=0; layer_idx<num_layers; layer_idx++) {
        m_layers.push_back(Layer());
    }

    // create Neurons in each layer
    // we do this as a separate loop so that we can pass pointers to prev and next layers to Neurons
    for (unsigned layer_idx=0; layer_idx<num_layers; layer_idx++) {
        debug_print("Layer %d\n", layer_idx);

        Layer *prev_layer = (layer_idx==0) ? NULL : &m_layers[layer_idx-1];
        Layer *next_layer = (layer_idx==num_layers-1) ? NULL : &m_layers[layer_idx+1];

        unsigned next_layer_size = (layer_idx==num_layers-1) ? 0 : layers[layer_idx+1];

        for (unsigned neur_idx=0; neur_idx<layers[layer_idx]; neur_idx++) {
            m_layers[layer_idx].push_back(Neuron(neur_idx, prev_layer, next_layer, next_layer_size));
            debug_print("\tCreated Neuron %d\n", neur_idx);
        }
        debug_print("\n");
    }
}


template <class N>
void Net<N>::feed_forward(const std::vector<double> &inp)
{
    debug_print("feed fwd\n");
    Layer &input_layer = m_layers.front();
    assert(inp.size() == input_layer.size()); // input vector should be of the same size as the num nodes in first layer

    for (unsigned n = 0; n < inp.size(); n++) { // actuate the first layer
        input_layer[n].set_value(inp[n]);
    }

    // activate all Neurons one layer at a time
    for (unsigned l = 0; l < m_layers.size(); l++) {
        for (unsigned n = 0; n < m_layers[l].size(); n++) {
            m_layers[l][n].activate();
        }
    }
}


template <class N>
void Net<N>::back_propagate_sgd(std::vector<double> &out)
{
    // performs isolated stochastic gradient decent
    // need to experiment with other methods

    debug_print("back prop\n");
    // calculate overall cost using sum of squared errors
    Layer &output_layer = m_layers.back();

    assert(out.size() == output_layer.size());

    m_avg_abs_error = 0;
    for (unsigned n = 0; n < output_layer.size(); n++) {
        double delta = out[n] - output_layer[n].get_value();
        m_error += delta;
        m_avg_abs_error += fabs(delta);
        // calculate output gradients while we're looping
        output_layer[n].calc_output_gradient(out[n]);
    }
    m_avg_abs_error /= output_layer.size(); //average error
    debug_print("\n");

    // calculate hidden gradients (starting at the last hidden layer, going to first)
    for (unsigned l = m_layers.size() - 2; l > 0; l-- ) {
        for (unsigned n = 0; n < m_layers[l].size(); n++) {
            m_layers[l][n].calc_hidden_gradient();
        }
        debug_print("\n");
    }

    // adjust previous layer weights going from last layer
    for (unsigned l = m_layers.size() - 1; l > 0; l-- ) {
        for (unsigned n = 0; n < m_layers[l].size(); n++) {
            m_layers[l][n].adjust_input_weights();
        }
        debug_print("\n");
    }
}



template <class N>
void Net<N>::get_results(std::vector<double> &results, double &avg_abs_error) const
{
    const Layer &output_layer = m_layers.back();
    for (unsigned n = 0; n < output_layer.size(); n++) {
        results.push_back(output_layer[n].get_value());
    }
    avg_abs_error = m_avg_abs_error;
}


template <class N>
void Net<N>::to_file(const std::string &filepath) const
{

}
