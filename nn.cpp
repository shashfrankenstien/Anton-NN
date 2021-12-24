#include <vector>
#include <stdlib.h> // srand and RAND_MAX
#include <time.h> // timestamp as seed for srand()
#include <stdio.h> // printf
#include <cassert> //assert

#include "nn.h"


#define debug_print(...) \
            do { if (PRINT_DEBUG_MSGS) printf( __VA_ARGS__); } while (0)

// ****************** Neuron ******************



Neuron::Neuron(unsigned idx, Layer *prev_layer, Layer *next_layer, unsigned next_layer_size)
: m_idx(idx),
m_prev_layer(prev_layer),
m_next_layer(next_layer),
m_activation_val(0)
{
    for (int i=0; i<next_layer_size; i++) {
        double w = INIT_WEIGHT_FUNC();
        m_conn_weights.push_back(w);
        m_old_conn_weight_deltas.push_back(0);
        debug_print("\t\tw: %f\n", w);
    }
}

void Neuron::set_value(double val)
{
    m_activation_val = val;
}

double Neuron::get_value() const
{
    return m_activation_val;
}


void Neuron::activate()
{
    if (m_prev_layer != NULL) {
        // combines neuron value with connection weight from neurons in the previous layer feeding current neuron
        m_sum_val = 0;
        for (unsigned n=0; n<m_prev_layer->size(); n++) {
            Neuron &prev_neuron = (*m_prev_layer)[n];
            m_sum_val += (prev_neuron.m_activation_val * prev_neuron.m_conn_weights[m_idx]);
        }
        // activate and set value!!
        m_activation_val = ACTIVATION_FUNC(m_sum_val - BIAS);
    }
}


void Neuron::calc_output_gradient(double target)
{
    // assuming our cost function is sum of squared errors,
    //  derivative of this wrt current neurons activation value will be -
    m_gradient = -2 * (target - m_activation_val) * ACTIVATION_DERIVATIVE_FUNC(m_activation_val);
    debug_print("\tg: %d - %f:%f\n", m_idx, m_activation_val, m_gradient);
}

void Neuron::calc_hidden_gradient()
{
    // to find how the hidden neuron influences the cost,
    //  we need to sum up all derivatives of weights going out of the neuron, times derivative of current activation value.
    //  this can be calculated by using the previously calculated gradients on the next layer.
    if (m_next_layer!=NULL) {
        double sum = 0;
        for (unsigned n=0; n<m_next_layer->size(); n++)
            sum += m_conn_weights[n] * (*m_next_layer)[n].m_gradient;

        m_gradient = sum * ACTIVATION_DERIVATIVE_FUNC(m_activation_val);
        debug_print("\tg: %d - %f:%f\n", m_idx, m_activation_val, m_gradient);
    }
}

void Neuron::adjust_input_weights()
{
    // adjust previous layer weights based on calculated gradient and learning rate
    // since we are using stochastic gradient decent, we add in some momentum using m_old_conn_weight_deltas to reduce noisy adjustments
    if (m_prev_layer!=NULL) {
        for (unsigned n=0; n<m_prev_layer->size(); n++) {
            Neuron &p_neuron = (*m_prev_layer)[n];
            double old_weight_delta = p_neuron.m_old_conn_weight_deltas[m_idx];
            // we're using learning rate, previous neuron activation and current gradient
            double new_delta_weight = (LEARNING_RATE * p_neuron.get_value() * m_gradient)
                                + (MOMENTUM_ALPHA * old_weight_delta); // include an additional factor in the direction of previous adjustment

            debug_print("\tadj: %dx%d - %f - (%f) = ", p_neuron.m_idx, m_idx, p_neuron.m_conn_weights[m_idx], new_delta_weight);
            p_neuron.m_conn_weights[m_idx] -= new_delta_weight;
            p_neuron.m_old_conn_weight_deltas[m_idx] = new_delta_weight;
            debug_print("%f\n", p_neuron.m_conn_weights[m_idx]);
        }
    }
}


// ****************** Net ******************


Net::Net(const std::vector<unsigned> &layers)
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


void Net::feed_forward(const std::vector<double> &inp)
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


void Net::back_propagate_sgd(std::vector<double> &out)
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
        m_avg_abs_error += abs(delta);
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



void Net::get_results(std::vector<double> &results, double &avg_abs_error) const
{
    const Layer &output_layer = m_layers.back();
    for (unsigned n = 0; n < output_layer.size(); n++) {
        results.push_back(output_layer[n].get_value());
    }
    avg_abs_error = m_avg_abs_error;
}


void Net::to_file(const std::string &filepath) const
{

}
