# Neural Network in C++

Based on the video tutorial by Dave Miller, information from 3Blue1Brown (Grant Sanderson) and StatQuest (Josh Starmer)
- https://www.youtube.com/watch?v=sK9AbJ4P8ao
- https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- https://www.youtube.com/watch?v=vMh0zPT0tLI&t=68s


---
Test with MNIST dataset

using https://github.com/wichtounet/mnist.git to parse binary files

---

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
