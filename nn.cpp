#include <vector>
#include <stdlib.h> // srand and RAND_MAX
#include <time.h> // timestamp as seed for srand()
#include <stdio.h> // printf
#include <cassert> //assert

#include "nn.h"


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
        printf("\t\tw: %f\n", w);
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

double Neuron::get_activation_for(unsigned neuron_idx) const
{
    // combines neuron value with connection weight for the neuron from the next layer requesting this value
    return m_activation_val * m_conn_weights[neuron_idx];
}


void Neuron::activate()
{
    if (m_prev_layer != NULL) {
        // grab values from previous layer
        m_sum_val = 0;
        for (unsigned n=0; n<m_prev_layer->size(); n++) {
            m_sum_val += (*m_prev_layer)[n].get_activation_for(m_idx);
        }
        // activate and set value!!
        m_activation_val = ACTIVATION_FUNC(m_sum_val - BIAS); // TODO: need to add some bias here
    }
}


void Neuron::calc_output_gradient(double target)
{
    // assuming our cost function is sum of squared errors,
    //  derivative of this wrt current neurons activation value will be -
    m_gradient = -2* (target - m_activation_val) * ACTIVATION_DERIVATIVE_FUNC(m_activation_val);
    printf("\tg: %d - %f:%f\n", m_idx, m_activation_val, m_gradient);
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
        printf("\tg: %d - %f:%f\n", m_idx, m_activation_val, m_gradient);
    }
}

void Neuron::adjust_prev_weights()
{
    // adjust previous layer weights based on calculated gradient and learning rate
    if (m_prev_layer!=NULL) {
        for (unsigned n=0; n<m_prev_layer->size(); n++) {
            Neuron &p_neuron = (*m_prev_layer)[n];

            // we're using learning rate, previous neuron activation and current gradient
            // think about momentum
            double delta_weight = LEARNING_RATE * p_neuron.get_value() * m_gradient;

            printf("\tadj: %dx%d - %f - (%f) = ", p_neuron.m_idx, m_idx, p_neuron.m_conn_weights[m_idx], delta_weight);
            p_neuron.m_conn_weights[m_idx] -= delta_weight;
            printf("%f\n", p_neuron.m_conn_weights[m_idx]);
        }
    }
}


// ****************** Net ******************


Net::Net(const std::vector<unsigned> &layers)
{
    unsigned num_layers = layers.size();

    // create Layers in each layer
    for (unsigned layer_idx=0; layer_idx<num_layers; layer_idx++) {
        m_layers.push_back(Layer());
    }

    // create Neurons in each layer
    // we do this as a separate loop so that we can pass pointers to prev and next layers to Neurons
    for (unsigned layer_idx=0; layer_idx<num_layers; layer_idx++) {
        printf("Layer %d\n", layer_idx);

        Layer *prev_layer = (layer_idx==0) ? NULL : &m_layers[layer_idx-1];
        Layer *next_layer = (layer_idx==num_layers-1) ? NULL : &m_layers[layer_idx+1];

        unsigned next_layer_size = (layer_idx==num_layers-1) ? 0 : layers[layer_idx+1];

        for (unsigned neur_idx=0; neur_idx<layers[layer_idx]; neur_idx++) {
            m_layers[layer_idx].push_back(Neuron(neur_idx, prev_layer, next_layer, next_layer_size));
            printf("\tCreated Neuron %d\n", neur_idx);
        }
        printf("\n");
    }
}


void Net::feed_forward(const std::vector<double> &inp)
{
    printf("feed fwd\n");
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


void Net::back_propagate(std::vector<double> &out)
{
    printf("back prop\n");
    // calculate overall cost using sum of squared errors
    Layer &output_layer = m_layers.back();

    assert(out.size() == output_layer.size());

    m_error = 0;
    for (unsigned n = 0; n < output_layer.size(); n++) {
        double delta = out[n] - output_layer[n].get_value();
        m_error += delta*delta;

        // calculate output gradients while we're looping
        output_layer[n].calc_output_gradient(out[n]);
    }
    printf("\n");

    // calculate hidden gradients (starting at the last hidden layer, going to first)
    for (unsigned l = m_layers.size() - 2; l > 0; l-- ) {
        for (unsigned n = 0; n < m_layers[l].size(); n++) {
            m_layers[l][n].calc_hidden_gradient();
        }
        printf("\n");
    }

    // adjust previous layer weights going from last layer
    for (unsigned l = m_layers.size() - 1; l > 0; l-- ) {
        for (unsigned n = 0; n < m_layers[l].size(); n++) {
            m_layers[l][n].adjust_prev_weights();
        }
        printf("\n");
    }
}


void Net::show() const
{
    printf("RES: ");
    const Layer &output_layer = m_layers.back();
    for (unsigned n = 0; n < output_layer.size(); n++) {
        printf("%f(%d) ", output_layer[n].get_value(), (output_layer[n].get_value()>=0.5) ? 1 : 0);
    }
    printf("ERR: %f\n", m_error);
}





// ************************************

// int main()
// {
//     srand(time(NULL));

//     std::vector<unsigned> layers{2, 5, 1}; // first layer has 3 Neuron, second has 2 and last has 1
//     Net myNet(layers);

//     std::vector<double> inputs{1,1};
//     myNet.feed_forward(inputs); // feed forward??

//     std::vector<double> outputs{0};
//     myNet.back_propagate(outputs);

//     myNet.save();

//     return 0;
// }
