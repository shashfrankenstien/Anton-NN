#include "conf.h"
#include "nn.h"



static unsigned calc_conv_output_dim(unsigned input_dim, unsigned kernel_size, unsigned padding, unsigned stride)
{
    if (kernel_size==0 || stride==0)
        return input_dim;
    return ((input_dim + (2*padding) - kernel_size) / stride) + 1;
}




ConvFrame::ConvFrame()
: rows(0),
columns(0),
m_data({})
{}

ConvFrame::ConvFrame(unsigned nrows, unsigned ncols)
: rows(nrows),
columns(ncols),
m_data(nrows*ncols, 0)
{
}

ConvFrame::ConvFrame(unsigned nrows, unsigned ncols, double init_val)
: rows(nrows),
columns(ncols),
m_data(nrows*ncols, init_val)
{
}

ConvFrame::ConvFrame(const ConvFrame& other)
: rows(other.rows),
columns(other.columns),
m_data(other.m_data)
{
}

ConvFrame::ConvFrame(const std::vector<std::vector<double>> &other)
: rows(other.size()),
columns(other.front().size()),
m_data(other.size()*other.front().size(), 0)
{
    for (unsigned r=0; r < rows; r++)
        for (unsigned c=0; c < columns; c++)
            operator()(r, c) = other[r][c];
}


void ConvFrame::reset_size(unsigned nrows, unsigned ncols)
{
    m_data.clear();
    m_data.resize(nrows*ncols, 0);
    rows = nrows;
    columns = ncols;
}

unsigned ConvFrame::calc_idx(unsigned row, unsigned col) const
{
    unsigned idx = (row * columns) + col;
    assert (idx < rows*columns);
    return idx;
}


double& ConvFrame::operator()(unsigned idx)
{
    assert (idx < rows*columns);
    return m_data[idx];
}

double& ConvFrame::operator()(unsigned row, unsigned col)
{
    unsigned idx = (row * columns) + col;
    return operator()(idx);
}

void ConvFrame::operator+=(const ConvFrame& other)
{
    assert(rows==other.rows && columns==other.columns);
    for (unsigned p = 0; p < rows * columns; p++)
        operator()(p) += other.get(p);
}

void ConvFrame::operator-=(const ConvFrame& other)
{
    assert(rows==other.rows && columns==other.columns);
    for (unsigned p = 0; p < rows * columns; p++)
        operator()(p) -= other.get(p);
}

void ConvFrame::operator+=(double scalar)
{
    for (unsigned p = 0; p < rows * columns; p++)
        operator()(p) += scalar;
}

void ConvFrame::operator-=(double scalar)
{
    for (unsigned p = 0; p < rows * columns; p++)
        operator()(p) -= scalar;
}



double ConvFrame::get(unsigned idx) const
{
    assert (idx < rows*columns);
    return m_data[idx];
}

double ConvFrame::get(unsigned row, unsigned col) const
{
    unsigned idx = (row * columns) + col;
    return get(idx);
}

double ConvFrame::min(unsigned strow, unsigned stcol, unsigned nrows, unsigned ncols) const
{
    double min_val = 99999;
    for (unsigned r = strow; r < strow+nrows; r++)
        for (unsigned c = stcol; c < stcol+ncols; c++)
            min_val = MIN(min_val, get(r, c));
    return min_val;
}

double ConvFrame::max(unsigned strow, unsigned stcol, unsigned nrows, unsigned ncols) const
{
    double max_val = -99999;
    for (unsigned r = strow; r < strow+nrows; r++)
        for (unsigned c = stcol; c < stcol+ncols; c++)
            max_val = MAX(max_val, get(r, c));
    return max_val;
}

double ConvFrame::sum(unsigned strow, unsigned stcol, unsigned nrows, unsigned ncols) const
{
    double sum_val = 0;
    for (unsigned r = strow; r < strow+nrows; r++)
        for (unsigned c = stcol; c < stcol+ncols; c++)
            sum_val += get(r, c);
    return sum_val;
}

double ConvFrame::avg(unsigned strow, unsigned stcol, unsigned nrows, unsigned ncols) const
{
    return sum(strow, stcol, nrows, ncols) / (double)(nrows * ncols);
}

double ConvFrame::min() const
{
    return min(0, 0, rows, columns);
}

double ConvFrame::max() const
{
    return max(0, 0, rows, columns);
}

double ConvFrame::sum() const
{
    return sum(0, 0, rows, columns);
}

double ConvFrame::avg() const
{
    return avg(0, 0, rows, columns);
}

/*
convolution involves multiplying values against the kernel and summing it to a single output cell
 - this implementation is lossy if stride and dimentions don't work out perfectly
 */
ConvFrame ConvFrame::convolve(const ConvFrame& kern, unsigned padding, unsigned stride) const
{

    ConvFrame out(
        calc_conv_output_dim(rows, kern.rows, padding, stride),
        calc_conv_output_dim(columns, kern.columns, padding, stride)
    );

    if (kern.rows==0 || kern.columns==0) { // if kernel size is 0, return 0s as output
        return out;
    }

    for (unsigned opr = 0; opr < out.rows; opr++) {
        for (unsigned opc = 0; opc < out.columns; opc++) {

            // these two loops apply the kernel and sum up the output at position opr, opc
            double sum_val = 0;
            for (unsigned kr = 0; kr < kern.rows; kr++) {
                for (unsigned kc = 0; kc < kern.columns; kc++) {
                    unsigned data_c = kc + (opc * stride);
                    unsigned data_r = kr + (opr * stride);
                    sum_val += (get(data_r, data_c) * kern.get(kr, kc));
                }
            }
            out(opr, opc) = sum_val;
        }
    }

    return out;
}


ConvFrame ConvFrame::max_pool(unsigned pool_factor, std::vector<unsigned>& pool_positions) const
{
    if (pool_factor <= 1) // can only perform pooling if the factor > 1
        return *this;

    unsigned pooled_rows = rows / pool_factor;
    unsigned pooled_columns = columns / pool_factor;

    // to be able to calculate pool positions, out indexing should be based on relevant dimentions
    //  - main reason to use orig_used_cols is for instances where odd frame sizes are pooled, which discards last column and / or last row
    unsigned orig_used_cols = pooled_columns * pool_factor;

    ConvFrame out(pooled_rows, pooled_columns);
    pool_positions.clear();
    pool_positions.resize(pooled_rows*pooled_columns, 0);

    for (unsigned opr = 0; opr < pooled_rows; opr++) {
        for (unsigned opc = 0; opc < pooled_columns; opc++) {

            // these two loops apply pooling based on pool_factor and assign the output at position opr, opc
            // note: we can't simply call max() here since we also need to keep track of pool positions for unpooling
            double max_val = -9999;
            unsigned pos = 0;
            for (unsigned fr = 0; fr < pool_factor; fr++) {
                for (unsigned fc = 0; fc < pool_factor; fc++) {
                    unsigned data_c = fc + (opc * pool_factor);
                    unsigned data_r = fr + (opr * pool_factor);
                    double data_val = get(data_r, data_c);
                    if (data_val > max_val) {
                        max_val = data_val;
                        pos = (data_r * orig_used_cols) + data_c;
                    }
                }
            }
            out(opr, opc) = max_val;
            pool_positions[out.calc_idx(opr, opc)] = pos;
        }
    }
    return out;
}


ConvFrame ConvFrame::max_unpool(unsigned pool_factor, std::vector<unsigned>& pool_positions) const
{
    assert(pool_positions.size()==rows*columns);

    ConvFrame out(rows * pool_factor, columns * pool_factor, 0);

    for (unsigned idx = 0; idx < rows*columns; idx++) {
        out(pool_positions[idx]) = get(idx);
    }
    return out;
}



ConvFrame ConvFrame::avg_pool(unsigned pool_factor) const
{
    if (pool_factor <= 1) // can only perform pooling if the factor > 1
        return *this;

    ConvFrame out(rows / pool_factor, columns / pool_factor);

    for (unsigned opr = 0; opr < out.rows; opr++) {
        for (unsigned opc = 0; opc < out.columns; opc++) {

            // we can call avg() here
            unsigned st_row = (opr * pool_factor);
            unsigned st_col = (opc * pool_factor);
            out(opr, opc) = avg(st_row, st_col, pool_factor, pool_factor);
        }
    }
    return out;
}




ConvFrame ConvFrame::avg_unpool(unsigned pool_factor) const
{
    ConvFrame out(rows * pool_factor, columns * pool_factor, 0);
    for (unsigned opr = 0; opr < out.rows; opr++) {
        for (unsigned opc = 0; opc < out.columns; opc++) {
            // floor division for position and distribute value
            out(opr, opc) = get(opr / pool_factor, opc / pool_factor) / (pool_factor * pool_factor);
        }
    }
    return out;
}

// apply a function onto all elements
void ConvFrame::map(std::function<double(double)> func)
{
    for (unsigned p = 0; p < rows * columns; p++)
        operator()(p) = func(get(p));
}


void ConvFrame::print()
{
    static double threshold = 0.5;
    for (unsigned r=0; r<rows; r++) {
        printf("|");
        for (unsigned c=0; c<columns; c++){
            #if 0
            printf("%.2f ", get(r,c));
            #else
            if (get(r, c)>threshold)
                printf("\u25A0 ");
            else if (get(r, c)< -threshold)
                printf(". ");
            else
                printf("  ");
            #endif
        }
        printf("|\n");
    }
    printf("\n");
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
        get_prev_layer_neuron(0).write_output_dimentions(m_in_dim);
    }

    // ammend input sizes if input pooling is enabled on this neuron
    if (kernel_conf.in_pooling != ConvTopology::NO_POOL && kernel_conf.in_pool_factor > 1) {
        m_in_dim[0] /= kernel_conf.in_pool_factor;
        m_in_dim[1] /= kernel_conf.in_pool_factor;
    }

    m_out_dim[0] = calc_conv_output_dim(
        m_in_dim[0], m_kernel_conf.kernel_size,
        m_kernel_conf.padding, m_kernel_conf.stride);
    m_out_dim[1] = calc_conv_output_dim(
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

void ConvNeuron::write_output_dimentions(unsigned (&dims)[2]) const
{
    dims[0] = m_out_dim[0];
    dims[1] = m_out_dim[1];
}

void ConvNeuron::write_input_dimentions(unsigned (&dims)[2]) const
{
    dims[0] = m_in_dim[0];
    dims[1] = m_in_dim[1];
}


// activated value has dimentions defined in m_in_dim
void ConvNeuron::set_activated_value(const ConvFrame &val)
{
    assert(val.rows==m_in_dim[0] && val.columns==m_in_dim[1]);

    m_activation_val = val;
}


double ConvNeuron::get_activated_value() const
{
    return m_activation_val.avg();
}

double ConvNeuron::get_error(double target) const
{
    return target - get_activated_value();
}


/*
called from next layer. uses the assigned kernel to produce activation value for that layer
- convolutions involve multiplying against the kernel and summing it to a single output cell
*/
ConvFrame ConvNeuron::get_conv_for(Neuron* other) const
{
    ConvNeuron* oth = (ConvNeuron*)other;
    const ConvFrame& kern = m_kernels[oth->m_idx];

    return m_activation_val.convolve(kern, m_kernel_conf.padding, m_kernel_conf.stride);
}


/*
combines neuron value with connection weight from neurons in the previous layer feeding current neuron
*/
void ConvNeuron::activate()
{
    if (m_prev_layer_size != 0) {
        debug_print("Activating Neuron %d\n", m_idx);

        // use output of first previous layer neuron to set up the dimentions of sum_val
        //  all other neurons in the layer will output the same dimentions
        ConvFrame sum_val = get_prev_layer_neuron(0).get_conv_for(this);
        for (unsigned n = 1; n < m_prev_layer_size; n++) {
            sum_val += get_prev_layer_neuron(n).get_conv_for(this);
        }

        // activate and set!
// feature to set output activation function separately :?
#ifdef OUTPUT_ACTIVATION_FUNC
        if (m_next_layer_size==0) // output layer
            sum_val.map([](double m){return OUTPUT_ACTIVATION_FUNC(m - BIAS);});
        else
            sum_val.map([](double m){return ACTIVATION_FUNC(m - BIAS);});
#else
        sum_val.map([](double m){return ACTIVATION_FUNC(m - BIAS);});
#endif // OUTPUT_ACTIVATION_FUNC

        sum_val.print();

        // check if pooling is required
        if (m_kernel_conf.in_pooling == ConvTopology::MAX_POOL && m_kernel_conf.in_pool_factor > 1) {
            debug_print("MAX POOL\n");
            set_activated_value(sum_val.max_pool(m_kernel_conf.in_pool_factor, m_max_pool_positions));
            m_activation_val.print();
            // m_activation_val.max_unpool(m_kernel_conf.in_pool_factor, m_max_pool_positions).print();
        }
        else if (m_kernel_conf.in_pooling == ConvTopology::AVG_POOL && m_kernel_conf.in_pool_factor > 1) {
            debug_print("AVG POOL\n");
            set_activated_value(sum_val.avg_pool(m_kernel_conf.in_pool_factor));
            m_activation_val.print();
            // m_activation_val.avg_unpool(m_kernel_conf.in_pool_factor).print();
        }
        else {
            debug_print("NO POOL\n");
            set_activated_value(sum_val);
        }
    }
}


/*
our current cost function is squared errors,
 derivative of this wrt current neurons activation value will be the output gradient

 - note: in this implementation:
        the gradient is actually only -2 * (target - act_val).
        but we multiply it with OUTPUT_ACTIVATION_DERIVATIVE_FUNC(act_val) to save some
        computation while calculating hidden gradients
*/
void ConvNeuron::calc_output_gradient(double target)
{
    double act_val = m_activation_val.avg();
#ifdef OUTPUT_ACTIVATION_DERIVATIVE_FUNC
    double out_grad = -2 * (target - act_val) * OUTPUT_ACTIVATION_DERIVATIVE_FUNC(act_val);
#else
    double out_grad = -2 * (target - act_val) * ACTIVATION_DERIVATIVE_FUNC(act_val);
#endif // OUTPUT_ACTIVATION_DERIVATIVE_FUNC

    // since act_val is the avg for now, we need to distribute the gradient evenly back
    double even_grad = out_grad / (m_activation_val.rows * m_activation_val.columns);
    m_gradient = ConvFrame(m_activation_val.rows, m_activation_val.columns, even_grad);
    m_gradient.print();
}


/*
to find how the hidden neuron influences the cost,
 we need to sum up all derivatives of weights going out of the neuron, times derivative of current activation value.
 this can be calculated by using the previously calculated gradients on the next layer.

 - note: in this implementation:
        the gradient is actually just the sum.
        but we multiply it with ACTIVATION_DERIVATIVE_FUNC(m_activation_val) to save some
        computation while calculating other hidden gradients
*/
void ConvNeuron::calc_hidden_gradient()
{
    // if (m_next_layer_size != 0) {
    //     double sum = 0;
    //     for (unsigned n=0; n<m_next_layer_size; n++) {
    //         sum += m_conn_weights[n] * get_next_layer_neuron(n).m_gradient;
    //     }
    //     m_gradient = sum * ACTIVATION_DERIVATIVE_FUNC(m_activation_val);
    //     debug_print("\tg: %d - %f:%f\n", m_idx, m_activation_val, m_gradient);
    // }
}


/*
adjust weights for other neuron on the next layer based on it's calculated gradient and learning rate
to obtain the adjustment to weights connecting to next layer neuron (other), we multiply current neuron activation, other gradient and learning rate
since we are using stochastic gradient decent, we add in some optional momentum using m_old_conn_weight_deltas to reduce noisy adjustments
*/
void ConvNeuron::adjust_weight_for(Neuron* other)
{
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




