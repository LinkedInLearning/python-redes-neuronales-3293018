import numpy as np

from activation import unit_step
from perceptron_monolayer import Perceptron


class MultiLayerPerceptron:

    def __init__(self, layers, bias=1.0, learning_rate=0.25):
        self.layers = np.array(layers)
        self.bias = bias
        self.learning_rate = learning_rate
        self.network = []
        self.values = []
        self.deltas = []

        for i in range(len(self.layers)):
            self.values.append([])
            self.deltas.append([])
            self.network.append([])
            self.values[i] = [0.0 for _ in range(self.layers[i])]
            self.deltas[i] = [0.0 for _ in range(self.layers[i])]
            if i > 0:
                for _ in range(self.layers[i]):
                    self.network[i].append(
                        Perceptron(
                            inputs=self.layers[i-1],
                            bias=self.bias,
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

    def print_network(self):
        print("---Network layers---")

        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print(f"Layer {i+1}, Neuron {j} weights: {self.network[i][j].weights}, bias: {self.network[i][j].bias}")

        print("------")

    def run(self, x):
        self.values[0] = x
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):  
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]

    def backprop(self, x, y):
        x = np.array(x, dtype=object)
        y = np.array(y, dtype=object)


        output = np.array(self.run(x))

        error = (y - output)
        mse = sum(error ** 2) / self.layers[-1]

        self.deltas[-1] = output * (1 - output) * (error)

        for i in reversed(range(1, len(self.network)-1)):
            for j in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]):
                    fwd_error += self.network[i+1][k].weights[j] * self.deltas[i+1][k]               
                self.deltas[i][j] = self.values[i][j] * (1-self.values[i][j]) * fwd_error

        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                for k in range(self.layers[i-1]):
                    delta_w = self.learning_rate * self.deltas[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] += delta_w

                delta_bias = self.learning_rate * self.deltas[i][j] * self.bias
                self.network[i][j].bias += delta_bias

        return mse
