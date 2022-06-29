import numpy as np
from random import random


def relu(x):
    return 0 if x <= 0 else 1


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def sigm(x):
    return 1.0 / (1 + np.exp(-x))


def lin(vals):
    return vals


def softmax(vals):
    sum_vals = sum([np.exp(v) for v in vals])
    o_vals = [np.exp(x) / sum_vals for x in vals]
    return o_vals


class Neuron:
    def __init__(self, prev_layer_size):
        self.weights = [random() for _ in range(prev_layer_size)]
        self.size = len(self.weights)
        self.bias = random()

    def activate(self, inp, activ_func):
        assert(len(inp) == len(self.weights))
        out = 0.0
        for i, w in zip(inp, self.weights):
            out += i*w
        return activ_func(out + self.bias)

    def _unpack(self, weights):
        for w in range(0, len(weights)-1):
            self.weights[w] = weights[w]
        self.bias = weights[-1]


class Layer:
    def __init__(self, size, prev_layer_size):
        self.neurons = [Neuron(prev_layer_size) for _ in range(size)]
        self.size = size

    def compute(self, inp, activ_func):
        out = []
        for neuron in self.neurons:
            out.append(neuron.activate(inp, activ_func))
        return out


class NeuralNetwork:
    def __init__(self, layer_sizes, hidden_activ_func=sigm, output_activ_func=lin):
        self.layers = []
        self.hidden_activ_func=hidden_activ_func
        self.output_activ_func=output_activ_func

        for i in range(1, len(layer_sizes)):
            if i == 1:  # First hidden layer
                self.layers.append(
                    Layer(layer_sizes[i], layer_sizes[i-1])
                )
            elif i == len(layer_sizes)-1:  # output layer
                self.layers.append(
                    Layer(layer_sizes[i], len(self.layers[i - 2].neurons))
                )
            else:
                self.layers.append(
                    Layer(layer_sizes[i], len(self.layers[i - 2].neurons))
                )

    def feed_forward(self, inp):
        """
        Run the Neural Network using a given input
        :param inp: List containing the inputs to the neural network
        :return: A list with the output of the Neural Network
        """
        vals = inp
        # Activate hidden layer
        for layer in range(len(self.layers) - 1):
            vals = self.layers[layer].compute(vals, self.hidden_activ_func)

        # Activate output layer
        vals = self.layers[-1].compute(vals, lin)

        return self.output_activ_func(vals)

    def pack(self):
        """
        Packs the Neural Network configuration into one long list of values
        :return: A list of weights + bias
        """
        out = []
        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    out.append(weight)
                out.append(neuron.bias)

        return out

    def unpack(self, weights):
        """
        Using a list of weights+bias, sets the neuronal networks weights to those values.
        :param weights: A list of weights + bias
        :return: None, This object has its weights set
        """
        w = weights[::-1]
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.weights = []
                for _ in range(neuron.size):
                    neuron.weights.append(w.pop())
                neuron.bias = w.pop()


if __name__ == '__main__':
    nn = NeuralNetwork([2, 10, 10, 2], hidden_activ_func=sigm, output_activ_func=lin)
    weights = nn.pack()
    print(nn.feed_forward([0.1, 0.2]))
    nn.unpack(weights)
    print(nn.feed_forward([0.1, 0.2]))