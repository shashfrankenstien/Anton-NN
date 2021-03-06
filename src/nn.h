#pragma once

#include <vector>
#include <stdio.h> // printf
#include <string>
#include <cassert> //assert
#include <cmath>
#include <type_traits> // std::is_same
#include <functional> // std::function

#ifndef PRINT_DEBUG_MSGS
#define PRINT_DEBUG_MSGS 0 // 1 or 0
#endif

#define debug_print(...) \
            do { if (PRINT_DEBUG_MSGS) printf( __VA_ARGS__); } while (0)


#define MIN(a, b) ((a<b)?a:b)
#define MAX(a, b) ((a>b)?a:b)


template <class N>
using Layer = std::vector<N>;


// ****************** Neuron ******************

/*
Basic Neuron class
*/
class Neuron
{
    public:
        Neuron(unsigned idx, Layer<Neuron>* prev_layer, unsigned prev_layer_size, Layer<Neuron>* next_layer, unsigned next_layer_size);

        virtual double get_activated_value() const;
        virtual void set_activated_value(double val);
        virtual double get_error(double target) const;

        virtual void activate(); // feed forward action
        virtual void adjust_input_weights(); // back prop action

        virtual void calc_output_gradient(double target);
        virtual void calc_hidden_gradient();

    protected:
        Layer<Neuron>* m_prev_layer;
        Layer<Neuron>* m_next_layer;
        unsigned m_prev_layer_size;
        unsigned m_next_layer_size;

        unsigned m_idx;
        double m_activation_val;
        double m_gradient;
        std::vector<double> m_conn_weights;
        std::vector<double> m_old_conn_weight_deltas;

        virtual double get_activation_for(Neuron* other) const;
        virtual void adjust_weight_for(Neuron* other);

        virtual Neuron& get_next_layer_neuron(unsigned other_idx);
        virtual Neuron& get_prev_layer_neuron(unsigned other_idx);
};


/*
Neuron with recurrent memory
*/
class RecurrentNeuron: public Neuron
{
    public:
        RecurrentNeuron(unsigned idx, Layer<RecurrentNeuron> *prev_layer, unsigned prev_layer_size, Layer<RecurrentNeuron> *next_layer, unsigned next_layer_size);

        void set_activated_value(double val) override;

    protected:
        double m_recur_activation_val;
        std::vector<double> m_recur_conn_weights;
        std::vector<double> m_recur_old_conn_weight_deltas;

        double get_activation_for(Neuron* other) const override;
        void adjust_weight_for(Neuron* other) override;

        RecurrentNeuron& get_next_layer_neuron(unsigned other_idx) override;
        RecurrentNeuron& get_prev_layer_neuron(unsigned other_idx) override;
};

/*
Input type for ConvNeuron based Net
*/
class ConvFrame
{
    public:
        unsigned rows;
        unsigned columns;

        ConvFrame();
        ConvFrame(unsigned nrows, unsigned ncols);
        ConvFrame(unsigned nrows, unsigned ncols, double init_val);
        ConvFrame(const ConvFrame& other);
        ConvFrame(const std::vector<std::vector<double>> &other);
        void reset_size(unsigned nrows, unsigned ncols);

        double& operator()(unsigned idx);
        double& operator()(unsigned row, unsigned col);
        void operator+=(const ConvFrame& other);
        void operator-=(const ConvFrame& other);
        void operator+=(double scalar);
        void operator-=(double scalar);
        // void operator*=(double scalar);

        double get(unsigned row, unsigned col) const;
        double get(unsigned idx) const;
        double min(unsigned strow, unsigned stcol, unsigned nrows, unsigned ncols) const;
        double max(unsigned strow, unsigned stcol, unsigned nrows, unsigned ncols) const;
        double sum(unsigned strow, unsigned stcol, unsigned nrows, unsigned ncols) const;
        double avg(unsigned strow, unsigned stcol, unsigned nrows, unsigned ncols) const;
        double min() const;
        double max() const;
        double sum() const;
        double avg() const;

        ConvFrame convolve(const ConvFrame& kern, unsigned padding, unsigned stride) const;

        ConvFrame max_pool(unsigned pool_factor, std::vector<unsigned>& pool_positions) const;
        ConvFrame max_unpool(unsigned pool_factor, std::vector<unsigned>& pool_positions) const;

        ConvFrame avg_pool(unsigned pool_factor) const;
        ConvFrame avg_unpool(unsigned pool_factor) const;

        // map a function on all values - used for activation
        void map(std::function<double(double)> func);

        void print();

    private:
        unsigned calc_idx(unsigned row, unsigned col) const;
        std::vector<double> m_data;
};

typedef struct {
    unsigned n_neurons;
    // input pooling
    enum : char {NO_POOL, MAX_POOL, AVG_POOL} in_pooling = NO_POOL;
    unsigned in_pool_factor = 2; // defaults to 2 because it's the most common
    // output convolutions
    unsigned kernel_size = 0;
    unsigned stride = 1;
    unsigned padding = 0;
} ConvTopology;


/*
convolutional neuron
- this is mostly a full rewrite of Neuron, but with similar interface
*/
class ConvNeuron: public Neuron
{
    public:
        ConvNeuron(unsigned idx, Layer<ConvNeuron> *prev_layer, unsigned prev_layer_size,
            Layer<ConvNeuron> *next_layer, unsigned next_layer_size,
            const ConvTopology &kernel_conf, const unsigned (&input_dimentions)[2]);

        void set_activated_value(const ConvFrame &val);
        double get_activated_value() const override;
        double get_error(double target) const override;

        void activate() override; // feed forward action
        void adjust_input_weights() override; // back prop action

        void calc_output_gradient(double target) override;
        void calc_hidden_gradient() override;

    protected:
        ConvTopology m_kernel_conf;
        unsigned m_in_dim[2] = {0,0};
        unsigned m_out_dim[2] = {0,0};
        ConvFrame m_activation_val;
        ConvFrame m_gradient;
        std::vector<ConvFrame> m_kernels;
        std::vector<ConvFrame> m_kernels_deltas;

        void write_output_dimentions(unsigned (&dims)[2]) const;
        void write_input_dimentions(unsigned (&dims)[2]) const;
        ConvFrame get_conv_for(Neuron* other) const;
        void adjust_weight_for(Neuron* other) override;

        ConvNeuron& get_next_layer_neuron(unsigned other_idx) override;
        ConvNeuron& get_prev_layer_neuron(unsigned other_idx) override;

        /* positions are stored if m_activation_val is pooled and needs to be unpooled
            ex: unpooling max pooled frame will involve
                - setting back the max value to the right position
                - setting other positions to 0
            however, average pooled m_activation_val will be divided across all postions while unpooling
            this variable is managed by ConvFrame when calling max_pool and max_unpool
        */
        std::vector<unsigned> m_max_pool_positions;

};


// ****************** Net ******************

/*
Neural Network template class
- N is the type of neuron to use
- I is the type of input. defaults to double
*/
template <class N, typename I = double>
class Net
{
    public:
        Net(const std::vector<unsigned> &layers);
        Net(const std::vector<ConvTopology> &topology, const unsigned (&input_dimentions)[2]); // convolutional net special case
        // ~Net();

        void feed_forward(const std::vector<I> &inp);
        void back_propagate_sgd(std::vector<double> &out);

        void get_results(std::vector<double> &results, double &avg_abs_error) const;
        void get_results(std::vector<double> &results, double &avg_abs_error, double threshold) const;

        void to_file(const std::string &filepath) const;

    private:
        std::vector<Layer<N>> m_layers;
        double m_error;
        double m_avg_abs_error;

};


/*
Constructor for simple and recurrent networs
*/
template <class N, typename I>
Net<N, I>::Net(const std::vector<unsigned> &layers)
: m_error(0)
{
    unsigned num_layers = layers.size();

    // create Layers in each layer
    for (unsigned layer_idx=0; layer_idx<num_layers; layer_idx++) {
        m_layers.push_back(Layer<N>());
    }

    // create Neurons in each layer
    // we do this as a separate loop so that we can pass pointers to prev and next layers to Neurons
    for (unsigned layer_idx=0; layer_idx<num_layers; layer_idx++) {
        debug_print("Layer<N> %d\n", layer_idx);

        Layer<N>* prev_layer = (layer_idx==0) ? NULL : &m_layers[layer_idx-1];
        unsigned prev_layer_size = (layer_idx==0) ? 0 : layers[layer_idx-1];

        Layer<N>* next_layer = (layer_idx==num_layers-1) ? NULL : &m_layers[layer_idx+1];
        unsigned next_layer_size = (layer_idx==num_layers-1) ? 0 : layers[layer_idx+1];

        for (unsigned neur_idx=0; neur_idx<layers[layer_idx]; neur_idx++) {
            m_layers[layer_idx].push_back(N(neur_idx, prev_layer, prev_layer_size, next_layer, next_layer_size));
            debug_print("\tCreated Neuron %d\n", neur_idx);
        }
        debug_print("\n");
    }
}

/*
Constructor for convolutional network
*/
template <class N, typename I>
Net<N, I>::Net(const std::vector<ConvTopology> &topology, const unsigned (&input_dimentions)[2])
: m_error(0)
{
    static_assert(std::is_same<ConvNeuron, N>::value && std::is_same<ConvFrame, I>::value, "Only ConvNeuron accepts ConvTopology constructor arg");
    unsigned num_layers = topology.size();

    // create Layers in each layer
    for (unsigned layer_idx=0; layer_idx<num_layers; layer_idx++) {
        m_layers.push_back(Layer<N>());
    }

    // create Neurons in each layer
    // we do this as a separate loop so that we can pass pointers to prev and next layers to Neurons
    for (unsigned layer_idx=0; layer_idx<num_layers; layer_idx++) {
        debug_print("Layer<N> %d\n", layer_idx);

        Layer<N>* prev_layer = (layer_idx==0) ? NULL : &m_layers[layer_idx-1];
        unsigned prev_layer_size = (layer_idx==0) ? 0 : topology[layer_idx-1].n_neurons;

        Layer<N>* next_layer = (layer_idx==num_layers-1) ? NULL : &m_layers[layer_idx+1];
        unsigned next_layer_size = (layer_idx==num_layers-1) ? 0 : topology[layer_idx+1].n_neurons;

        for (unsigned neur_idx=0; neur_idx<topology[layer_idx].n_neurons; neur_idx++) {
            m_layers[layer_idx].push_back(N(
                neur_idx, prev_layer, prev_layer_size,
                next_layer, next_layer_size,
                topology[layer_idx], input_dimentions
            ));
            debug_print("\tCreated Neuron %d\n", neur_idx);
        }
        debug_print("\n");
    }
}


/*
input vector should be of the same size as the num nodes in first layer
*/
template <class N, typename I>
void Net<N, I>::feed_forward(const std::vector<I> &inp)
{
    debug_print("feed fwd\n");
    Layer<N> &input_layer = m_layers.front();
    assert(inp.size() == input_layer.size());

    for (unsigned n = 0; n < inp.size(); n++) { // actuate the first layer
        input_layer[n].set_activated_value(inp[n]);
    }

    // activate all Neurons one layer at a time
    for (unsigned l = 0; l < m_layers.size(); l++) {
        for (unsigned n = 0; n < m_layers[l].size(); n++) {
            m_layers[l][n].activate();
        }
    }
}

/*
performs isolated stochastic gradient decent
need to experiment with other methods
*/
template <class N, typename I>
void Net<N, I>::back_propagate_sgd(std::vector<double> &out)
{
    debug_print("back prop\n");
    // calculate overall cost using sum of squared errors
    Layer<N> &output_layer = m_layers.back();

    assert(out.size() == output_layer.size());

    m_avg_abs_error = 0;
    for (unsigned n = 0; n < output_layer.size(); n++) {
        double delta = output_layer[n].get_error(out[n]);
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



template <class N, typename I>
void Net<N, I>::get_results(std::vector<double> &results, double &avg_abs_error) const
{
    const Layer<N> &output_layer = m_layers.back();
    for (unsigned n = 0; n < output_layer.size(); n++) {
        results.push_back(output_layer[n].get_activated_value());
    }
    avg_abs_error = m_avg_abs_error;
}

template <class N, typename I>
void Net<N, I>::get_results(std::vector<double> &results, double &avg_abs_error, double threshold) const
{
    const Layer<N> &output_layer = m_layers.back();
    for (unsigned n = 0; n < output_layer.size(); n++) {
        results.push_back((output_layer[n].get_activated_value() >= threshold) ? 1 : 0);
    }
    avg_abs_error = m_avg_abs_error;
}


template <class N, typename I>
void Net<N, I>::to_file(const std::string &filepath) const
{

}
