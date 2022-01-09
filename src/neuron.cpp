#include "conf.h"
#include "nn.h"

// ****************** Basic Neuron ******************

Neuron::Neuron(unsigned idx, Layer<Neuron>* prev_layer, unsigned prev_layer_size, Layer<Neuron>* next_layer, unsigned next_layer_size)
: m_idx(idx),
m_prev_layer(prev_layer),
m_prev_layer_size(prev_layer_size),
m_next_layer(next_layer),
m_next_layer_size(next_layer_size),
m_activation_val(0),
m_gradient(0)
{
    for (int i=0; i<next_layer_size; i++) {
        double w = INIT_WEIGHT_FUNC();
        m_conn_weights.push_back(w);
        m_old_conn_weight_deltas.push_back(0);
        debug_print("\t\tw: %f\n", w);
    }
}

Neuron& Neuron::get_next_layer_neuron(unsigned other_idx)
{
    return (*m_next_layer)[other_idx];
}

Neuron& Neuron::get_prev_layer_neuron(unsigned other_idx)
{
    return (*m_prev_layer)[other_idx];
}

void Neuron::set_activated_value(double val)
{
    m_activation_val = val;
}

double Neuron::get_activated_value() const
{
    return m_activation_val;
}

double Neuron::get_error(double target) const
{
    return target - get_activated_value();
}


double Neuron::get_activation_for(Neuron* other) const
{
    return (m_activation_val * m_conn_weights[other->m_idx]);
}

/*
combines neuron value with connection weight from neurons in the previous layer feeding current neuron
*/
void Neuron::activate()
{
    if (m_prev_layer_size != 0) {
        double sum_val = 0;
        for (unsigned n=0; n<m_prev_layer_size; n++) {
            sum_val += get_prev_layer_neuron(n).get_activation_for(this);
        }
        // activate and set value!!
// feature to set output activation function separately :?
#ifdef OUTPUT_ACTIVATION_FUNC
        if (m_next_layer_size==0) // output layer
            set_activated_value(OUTPUT_ACTIVATION_FUNC(sum_val - BIAS));
        else
            set_activated_value(ACTIVATION_FUNC(sum_val - BIAS));
#else
        set_activated_value(ACTIVATION_FUNC(sum_val - BIAS));
#endif // OUTPUT_ACTIVATION_FUNC

    }
}

/*
adjust weights for other neuron on the next layer based on it's calculated gradient and learning rate
since we are using stochastic gradient decent, we add in some optional momentum using m_old_conn_weight_deltas to reduce noisy adjustments
we're using learning rate, previous neuron activation and current gradient
*/
void Neuron::adjust_weight_for(Neuron* other)
{

    double old_weight_delta = m_old_conn_weight_deltas[other->m_idx];
    // we're using learning rate, previous neuron activation and current gradient
    double new_delta_weight = (LEARNING_RATE * m_activation_val * other->m_gradient)
                        + (MOMENTUM_ALPHA * old_weight_delta); // include an additional factor in the direction of previous adjustment

    debug_print("\tadj: %dx%d - %f - (%f) = ", m_idx, other->m_idx, m_conn_weights[other->m_idx], new_delta_weight);
    m_conn_weights[other->m_idx] -= new_delta_weight;
    m_old_conn_weight_deltas[other->m_idx] = new_delta_weight;
    debug_print("%f\n", m_conn_weights[other->m_idx]);
}


void Neuron::adjust_input_weights()
{
    if (m_prev_layer_size != 0) {
        for (unsigned n=0; n<m_prev_layer_size; n++) {
            get_prev_layer_neuron(n).adjust_weight_for(this);
        }
    }
}

/*
our current cost function is sum of squared errors,
 derivative of this wrt current neurons activation value will be the output gradient
*/
void Neuron::calc_output_gradient(double target)
{
#ifdef OUTPUT_ACTIVATION_DERIVATIVE_FUNC
    m_gradient = -2 * (target - m_activation_val) * OUTPUT_ACTIVATION_DERIVATIVE_FUNC(m_activation_val);
#else
    m_gradient = -2 * (target - m_activation_val) * ACTIVATION_DERIVATIVE_FUNC(m_activation_val);
#endif // OUTPUT_ACTIVATION_DERIVATIVE_FUNC
    debug_print("\tg: %d - %f:%f\n", m_idx, m_activation_val, m_gradient);
}

/*
to find how the hidden neuron influences the cost,
 we need to sum up all derivatives of weights going out of the neuron, times derivative of current activation value.
 this can be calculated by using the previously calculated gradients on the next layer.
*/
void Neuron::calc_hidden_gradient()
{
    if (m_next_layer_size != 0) {
        double sum = 0;
        for (unsigned n=0; n<m_next_layer_size; n++) {
            sum += m_conn_weights[n] * get_next_layer_neuron(n).m_gradient;
        }
        m_gradient = sum * ACTIVATION_DERIVATIVE_FUNC(m_activation_val);
        debug_print("\tg: %d - %f:%f\n", m_idx, m_activation_val, m_gradient);
    }
}






// ****************** Recurrent Neuron ******************



RecurrentNeuron::RecurrentNeuron(unsigned idx, Layer<RecurrentNeuron> *prev_layer, unsigned prev_layer_size, Layer<RecurrentNeuron> *next_layer, unsigned next_layer_size)
: Neuron(idx, (Layer<Neuron>*)prev_layer, prev_layer_size, (Layer<Neuron>*)next_layer, next_layer_size),
m_recur_activation_val(0)
{
    for (int i=0; i<next_layer_size; i++) {
        double w = INIT_WEIGHT_FUNC();
        m_recur_conn_weights.push_back(w);
        m_recur_old_conn_weight_deltas.push_back(0);
        debug_print("\t\tr_w: %f\n", w);
    }
}
RecurrentNeuron& RecurrentNeuron::get_next_layer_neuron(unsigned other_idx)
{
    return (*(Layer<RecurrentNeuron>*)m_next_layer)[other_idx];
}
RecurrentNeuron& RecurrentNeuron::get_prev_layer_neuron(unsigned other_idx)
{
    return (*(Layer<RecurrentNeuron>*)m_prev_layer)[other_idx];
}

void RecurrentNeuron::set_activated_value(double val)
{
    m_recur_activation_val = m_activation_val;
    m_activation_val = val;
}


double RecurrentNeuron::get_activation_for(Neuron* other) const
{
    RecurrentNeuron* oth = (RecurrentNeuron*)other;
    return (m_activation_val * m_conn_weights[oth->m_idx]) + (m_recur_activation_val * m_recur_conn_weights[oth->m_idx]);
}


/*
adjust weights for other neuron on the next layer based on it's calculated gradient and learning rate
since we are using stochastic gradient decent, we add in some optional momentum using m_old_conn_weight_deltas to reduce noisy adjustments
we're using learning rate, previous neuron activation and current gradient

this method override does this for the recurrent cell of the neuron
*/
void RecurrentNeuron::adjust_weight_for(Neuron* other)
{
    Neuron::adjust_weight_for(other);

    RecurrentNeuron* oth = (RecurrentNeuron*)other;
    // now doing the same for recurrent part of the neuron
    double old_weight_delta = m_recur_old_conn_weight_deltas[oth->m_idx];
    // we're using learning rate, previous neuron activation and current gradient
    double new_delta_weight = (LEARNING_RATE * m_recur_activation_val * oth->m_gradient)
                        + (MOMENTUM_ALPHA * old_weight_delta); // include an additional factor in the direction of previous adjustment

    debug_print("\tadj: %dx%d - %f - (%f) = ", m_idx, oth->m_idx, m_recur_conn_weights[oth->m_idx], new_delta_weight);
    m_recur_conn_weights[oth->m_idx] -= new_delta_weight;
    m_recur_old_conn_weight_deltas[oth->m_idx] = new_delta_weight;
    debug_print("%f\n", m_recur_conn_weights[oth->m_idx]);
}
