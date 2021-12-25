#include "conf.h"
#include "nn.h"


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

double Neuron::get_activation_for(unsigned other_idx) const
{
    return (m_activation_val * m_conn_weights[other_idx]);
}

void Neuron::adjust_weight_for(unsigned other_idx, double other_gradient)
{
    // adjust weights for other neuron on the next layer based on it's calculated gradient and learning rate
    // since we are using stochastic gradient decent, we add in some momentum using m_old_conn_weight_deltas to reduce noisy adjustments
    double old_weight_delta = m_old_conn_weight_deltas[other_idx];
    // we're using learning rate, previous neuron activation and current gradient
    double new_delta_weight = (LEARNING_RATE * m_activation_val * other_gradient)
                        + (MOMENTUM_ALPHA * old_weight_delta); // include an additional factor in the direction of previous adjustment

    debug_print("\tadj: %dx%d - %f - (%f) = ", m_idx, other_idx, m_conn_weights[other_idx], new_delta_weight);
    m_conn_weights[other_idx] -= new_delta_weight;
    m_old_conn_weight_deltas[other_idx] = new_delta_weight;
    debug_print("%f\n", m_conn_weights[other_idx]);
}


void Neuron::activate()
{
    if (m_prev_layer != NULL) {
        // combines neuron value with connection weight from neurons in the previous layer feeding current neuron
        double sum_val = 0;
        for (unsigned n=0; n<m_prev_layer->size(); n++) {
            Neuron &prev_neuron = (*m_prev_layer)[n];
            sum_val += prev_neuron.get_activation_for(m_idx);
        }
        // activate and set value!!
        m_activation_val = ACTIVATION_FUNC(sum_val - BIAS);
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
    if (m_prev_layer!=NULL) {
        for (unsigned n=0; n<m_prev_layer->size(); n++) {
            (*m_prev_layer)[n].adjust_weight_for(m_idx, m_gradient);
        }
    }
}
