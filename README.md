# Neural Network (Hardcoded)

This document provides an overview of a simple neural network implemented in Python using the `numpy` library. The network consists of a customizable number of layers, each with a specified number of input and output neurons. The network is designed to be a feedforward neural network, and it includes a basic implementation of the forward pass.

## Class: `Layer`

The `Layer` class represents a single layer in the neural network. Each layer is initialized with the number of input and output nodes. The weights of the connections between nodes are randomly initialized, and an activation function can be applied to each neuron. The available activation functions are sigmoid, ReLU, and hyperbolic tangent (tanh).

### Methods:

- **`__init__(self, inNodes, outNodes)`**: Initialize the layer with the given number of input and output nodes. Randomly initialize weights and set the activation function to zero.

- **`desc_layer(self)`**: Display information about the layer, including the number of input and output nodes, weights, available activation functions, and the current activation stack.

- **`activation_stack(self, activation_stack)`**: Set the activation stack, specifying which activation function to apply to each neuron.

- **`activationFunc(self, output)`**: Apply the specified activation function to the output of the layer.

- **`fwd(self, input)`**: Perform the forward pass through the layer. Add bias to the input, calculate the sum of inputs, apply activation function, and return the output.

## Class: `NeuralNetwork`

The `NeuralNetwork` class represents the entire neural network, composed of multiple layers. It includes methods for defining weights, describing the network, setting activation functions for layers, evaluating input vectors, and training the network.

### Methods:

- **`__init__(self, features, hiddenNeurons, classes, hiddenLayers=1)`**: Initialize the neural network with the specified number of input features, hidden neurons, output classes, and optional hidden layers.

- **`def_weight(self, matrix, hiddenlayer=1)`**: Define weights for a specific hidden layer.

- **`desc_network(self)`**: Display information about each layer in the network.

- **`def_layer_activation(self, stack, layer)`**: Set the activation stack for a specific layer.

- **`eval(self, input)`**: Perform the forward pass through the entire network and return the final output.

- **`train(self, input, target, iterations=10000)`**: Train the network by performing the forward pass for each input vector and displaying the output. This method is intended for basic demonstration purposes.

## Example Usage:

```python
# Create a neural network with 2 input features, 3 hidden neurons, and 2 output classes
nn = NeuralNetwork(features=2, hiddenNeurons=3, classes=2, hiddenLayers=1)

# Describe the network
nn.desc_network()

# Set activation functions for layers
nn.def_layer_activation(stack=[0, 1, 2], layer=0)
nn.def_layer_activation(stack=[1], layer=1)

# Evaluate an input vector
input_vector = np.array([[0.5, 0.8]])
output = nn.eval(input_vector)
print(f"Output for input vector: {output}")

# Train the network with input and target data
input_data = np.array([[0.1, 0.2], [0.3, 0.4]])
target_data = np.array([[0, 1], [1, 0]])
nn.train(input_data, target_data, iterations=1000)
```

Please note that this implementation is a basic demonstration and may not be suitable for real-world applications. It is recommended to use established machine learning libraries like TensorFlow or PyTorch for more complex neural network tasks.
