#pragma once

#include <stdlib.h> // srand, rand and RAND_MAX
#include <algorithm> // std::max
#include <cmath>

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
    return std::max(0.0, z);
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


inline double random_weights()
{
    int r = rand();
    return (((double)r / (double)RAND_MAX) * 2) - 1; // random number between -1 and 1
}

inline double large_random_weights()
{
    int r = rand();
    return (((double)r / (double)RAND_MAX) * 8) - 4; // random number between -4 and 4
}


// TODO: add cost function option and derivative of it here

#define INIT_WEIGHT_FUNC large_random_weights
#define ACTIVATION_FUNC Sigmoid
#define ACTIVATION_DERIVATIVE_FUNC DSigmoid

#define RANDOM_SEED 0
#define BIAS 0.5
#define LEARNING_RATE 0.1
#define MOMENTUM_ALPHA 0
