#include "conf.h"
#include "nn.h"


ConvFrame::ConvFrame(){}

ConvFrame::ConvFrame(unsigned nrows, unsigned ncols)
: m_rows(nrows),
m_cols(ncols),
m_data(nrows*ncols, 0)
{
}

ConvFrame::ConvFrame(const ConvFrame& other)
: m_rows(other.m_rows),
m_cols(other.m_cols),
m_data(other.m_data)
{
}

ConvFrame::ConvFrame(const std::vector<std::vector<double>> &other)
: m_rows(other.size()),
m_cols(other.front().size()),
m_data(other.size()*other.front().size(), 0)
{
    for (unsigned r=0; r < m_rows; r++)
        for (unsigned c=0; c < m_cols; c++)
            operator()(r, c) = other[r][c];
}


void ConvFrame::reset_size(unsigned nrows, unsigned ncols)
{
    m_data.clear();
    m_data.resize(nrows*ncols, 0);
    m_rows = nrows;
    m_cols = ncols;
}


double ConvFrame::get(unsigned idx) const
{
    assert (idx < m_rows*m_cols);
    return m_data[idx];
}

double ConvFrame::get(unsigned row, unsigned col) const
{
    unsigned idx = (row * m_cols) + col;
    return get(idx);
}

double ConvFrame::avg() const
{
    double sum = 0;
    for (unsigned p = 0; p < m_rows * m_cols; p++)
        sum += get(p);
    return sum / (double)(m_rows * m_cols);
}

double& ConvFrame::operator()(unsigned idx)
{
    assert (idx < m_rows*m_cols);
    return m_data[idx];
}

double& ConvFrame::operator()(unsigned row, unsigned col)
{
    unsigned idx = (row * m_cols) + col;
    return operator()(idx);
}

void ConvFrame::operator+=(const ConvFrame& other)
{
    assert(m_rows==other.m_rows && m_cols==other.m_cols);
    for (unsigned p = 0; p < m_rows * m_cols; p++)
        operator()(p) += other.get(p);
}

void ConvFrame::operator-=(const ConvFrame& other)
{
    assert(m_rows==other.m_rows && m_cols==other.m_cols);
    for (unsigned p = 0; p < m_rows * m_cols; p++)
        operator()(p) -= other.get(p);
}

void ConvFrame::operator+=(double scalar)
{
    for (unsigned p = 0; p < m_rows * m_cols; p++)
        operator()(p) += scalar;
}

void ConvFrame::operator-=(double scalar)
{
    for (unsigned p = 0; p < m_rows * m_cols; p++)
        operator()(p) -= scalar;
}

void ConvFrame::print()
{
    static double threshold = 0.4;
    for (unsigned r=0; r<m_rows; r++) {
        printf("|");
        for (unsigned c=0; c<m_cols; c++){
            // printf("%.2f ", get(r,c));
            if (get(r, c)>threshold)
                printf("\u25A0 ");
            else if (get(r, c)< -threshold)
                printf(". ");
            else
                printf("  ");
        }
        printf("|\n");
    }
    printf("\n");
}




static unsigned calc_output_dim(unsigned input_dim, unsigned kernel_size, unsigned padding, unsigned stride)
{
    if (kernel_size==0 || stride==0)
        return input_dim;
    return ((input_dim + (2*padding) - kernel_size) / stride) + 1;
}



// ****************** Convolutional Neuron ******************


ConvNeuron::ConvNeuron(unsigned idx, Layer<ConvNeuron>* prev_layer, unsigned prev_layer_size,
    Layer<ConvNeuron>* next_layer, unsigned next_layer_size,
    const ConvTopology &kernel_conf, const unsigned (&input_dimentions)[2])
: Neuron(idx, (Layer<Neuron>*)prev_layer, prev_layer_size, (Layer<Neuron>*)next_layer, next_layer_size),
m_kernel_conf(kernel_conf)
{
    for (int i=0; i<next_layer_size; i++) {
        ConvFrame new_kernel(kernel_conf.kernel_size, kernel_conf.kernel_size);
        ConvFrame old_deltas(kernel_conf.kernel_size, kernel_conf.kernel_size);
        for (unsigned i = 0; i < kernel_conf.kernel_size * kernel_conf.kernel_size; i++) {
            new_kernel(i) = INIT_WEIGHT_FUNC();
        }
        new_kernel.print();
        m_kernels.push_back(new_kernel);
        m_kernels_deltas.push_back(old_deltas);
    }

    if (prev_layer_size==0) { // input layer
        m_in_dim[0] = input_dimentions[0];
        m_in_dim[1] = input_dimentions[1];
    } else {
        get_prev_layer_neuron(0).get_output_dimentions(m_in_dim);
    }
    m_out_dim[0] = calc_output_dim(
        m_in_dim[0], m_kernel_conf.kernel_size,
        m_kernel_conf.padding, m_kernel_conf.stride);
    m_out_dim[1] = calc_output_dim(
        m_in_dim[1], m_kernel_conf.kernel_size,
        m_kernel_conf.padding, m_kernel_conf.stride);

    debug_print("%d x %d -> %d x %d", m_in_dim[0], m_in_dim[1], m_out_dim[0], m_out_dim[1]);
}

ConvNeuron& ConvNeuron::get_next_layer_neuron(unsigned other_idx)
{
    return (*(Layer<ConvNeuron>*)m_next_layer)[other_idx];
}

ConvNeuron& ConvNeuron::get_prev_layer_neuron(unsigned other_idx)
{
    return (*(Layer<ConvNeuron>*)m_prev_layer)[other_idx];
}

void ConvNeuron::get_output_dimentions(unsigned (&dims)[2]) const
{
    dims[0] = m_out_dim[0];
    dims[1] = m_out_dim[1];
}

void ConvNeuron::get_input_dimentions(unsigned (&dims)[2]) const
{
    dims[0] = m_in_dim[0];
    dims[1] = m_in_dim[1];
}

void ConvNeuron::set_value(const ConvFrame &val)
{
    assert(val.m_rows==m_in_dim[0] && val.m_cols==m_in_dim[1]);
    m_activation_val = val;
}

double ConvNeuron::get_value() const
{
    return ACTIVATION_FUNC(m_activation_val.avg());
}

double ConvNeuron::get_error(double out) const
{
    return out - get_value();
}


// called from next layer. uses the assigned kernel to produce activation value for that layer
// - convolutions involve multiplying against the kernel and averaging it to a single output cell
ConvFrame ConvNeuron::get_conv_for(Neuron* other) const
{
    ConvNeuron* oth = (ConvNeuron*)other;
    const ConvFrame& kern = m_kernels[oth->m_idx];

    if (kern.m_rows==0) {
        return oth->m_activation_val;
    }

    ConvFrame out(m_out_dim[0], m_out_dim[1]);

    // convolution involve multiplying against the kernel and averaging it to a single output cell
    unsigned count = 0;
    for (unsigned opr = 0; opr < m_out_dim[0]; opr++) {
        for (unsigned opc = 0; opc < m_out_dim[1]; opc++) {

            double sum = 0;
            // these two loops apply the kernel at position opr, opc
            for (unsigned kr = 0; kr < kern.m_rows; kr++) {
                for (unsigned kc = 0; kc < kern.m_cols; kc+=m_kernel_conf.stride) {
                    unsigned img_c = opc+kc;
                    unsigned img_r = opr+kr;
                    sum += (m_activation_val.get(img_r, img_c) * kern.get(kr, kc));
                }
            }
            out(opr, opc) = sum; // - BIAS; // / (kern.m_cols*kern.m_rows);
        }
    }

    return out;
}


// combines neuron value with connection weight from neurons in the previous layer feeding current neuron
void ConvNeuron::activate()
{
    if (m_prev_layer_size != 0) {
        debug_print("Activating Neuron %d\n", m_idx);
        ConvFrame sum_val(m_in_dim[0], m_in_dim[1]);
        for (unsigned n=0; n<m_prev_layer_size; n++) {
            sum_val += get_prev_layer_neuron(n).get_conv_for(this);
        }
        // activate and set value!!
        sum_val.print();
        set_value(sum_val);
    }
}



void ConvNeuron::adjust_weight_for(Neuron* other)
{
    // // adjust weights for other neuron on the next layer based on it's calculated gradient and learning rate
    // // since we are using stochastic gradient decent, we add in some momentum using m_old_conn_weight_deltas to reduce noisy adjustments
    // ConvNeuron* oth = (ConvNeuron*)other;
    // double old_weight_delta = m_old_conn_weight_deltas[oth->m_idx];
    // // we're using learning rate, previous neuron activation and current gradient
    // double new_delta_weight = (LEARNING_RATE * m_activation_val * oth->m_gradient)
    //                     + (MOMENTUM_ALPHA * old_weight_delta); // include an additional factor in the direction of previous adjustment

    // debug_print("\tadj: %dx%d - %f - (%f) = ", m_idx, oth->m_idx, m_conn_weights[oth->m_idx], new_delta_weight);
    // m_conn_weights[oth->m_idx] -= new_delta_weight;
    // m_old_conn_weight_deltas[oth->m_idx] = new_delta_weight;
    // debug_print("%f\n", m_conn_weights[oth->m_idx]);
}


void ConvNeuron::adjust_input_weights()
{
    if (m_prev_layer_size != 0) {
        for (unsigned n=0; n<m_prev_layer_size; n++) {
            get_prev_layer_neuron(n).adjust_weight_for(this);
        }
    }
}


void ConvNeuron::calc_output_gradient(double target)
{
    // // assuming our cost function is sum of squared errors,
    // //  derivative of this wrt current neurons activation value will be -
    // m_gradient = -2 * (target - m_activation_val) * ACTIVATION_DERIVATIVE_FUNC(m_activation_val);
    // debug_print("\tg: %d - %f:%f\n", m_idx, m_activation_val, m_gradient);
}

void ConvNeuron::calc_hidden_gradient()
{
    // // to find how the hidden neuron influences the cost,
    // //  we need to sum up all derivatives of weights going out of the neuron, times derivative of current activation value.
    // //  this can be calculated by using the previously calculated gradients on the next layer.
    // if (m_next_layer_size != 0) {
    //     double sum = 0;
    //     for (unsigned n=0; n<m_next_layer_size; n++) {
    //         sum += m_conn_weights[n] * get_next_layer_neuron(n).m_gradient;
    //     }
    //     m_gradient = sum * ACTIVATION_DERIVATIVE_FUNC(m_activation_val);
    //     debug_print("\tg: %d - %f:%f\n", m_idx, m_activation_val, m_gradient);
    // }
}


