import numpy as np

from activation import unit_step
from perceptron_monolayer import Perceptron


class MultiLayerPerceptron:

    def __init__(self, layers, bias=1.0):
        self.layers = np.array(layers)
        self.bias = bias
        self.network = []
        self.values = []

        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for _ in range(self.layers[i])]
            if i > 0:
                for j in range(self.layers[i]):
                    self.network[i].append(
                        Perceptron(
                            inputs=self.layers[i-1],
                            bias=self.bias,
                            activation_function=unit_step,
                        )
                    )

    def set_weights(self, w_init):
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])

    def set_bias(self, b_init):
        for i in range(len(b_init)):
            for j in range(len(b_init[i])):
                self.network[i+1][j].set_bias(b_init[i][j])

    def run(self, x):
        self.values[0] = x
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):  
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]

    def print_network(self):
        print("---Network layers---")

        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print(f"Layer {i+1}, Neuron {j} weights: {self.network[i][j].weights}, bias: {self.network[i][j].bias}")

        print("------")
