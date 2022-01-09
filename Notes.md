# Notes :notebook_with_decorative_cover:

## Observations

### Activation functions
- Sigmoid activation function is generally very good

- TanH activation function works well with simpler networks, but strugles with complexity
    - In mnist data, it's hitting a cap of 50% moving average success rate for some reason
    - RNN performance is also pretty bad
    - Maybe better for CNN

- ReLU
    - Maybe better for CNN

(These are probably issues with this implementation since it was built using sigmoid, and others were tried to be forced in)

### Learning rate
- generally 0.1 is a good value
- recurrent net performs better with 0.05 - 0.08 learning rates


## TODO

- CNN input topology validation (because overflows are not handled yet. assert gets called)
- remove all asserts
