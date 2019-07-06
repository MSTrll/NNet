from Networks import Functions as FC
import numpy as np
import random


class Network:

    def quadratic_derivative(self, output_activations, y):
        return (output_activations - y)

    def cross_entropy_derivative(self, output_activations, y):
        return (1 - y)/(1 - output_activations) - y/output_activations

    def __init__(self, sizes, outp_function=quadratic_derivative, output=True):

        self.outp_function = outp_function

        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.activations = [np.zeros([y, 1]) for y in sizes]
        self.output = output

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data is not None: n_test = len(test_data)
        n = len(training_data)
        success_tests = 0
        for ep in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None and self.output:
                success_tests = self.evaluate(test_data)
                print("Epoch {0}: {1}".format(
                    ep, success_tests/n_test))
            elif self.output:
                print("Epoch {0} completed".format(ep))
        if test_data is not None:
            return success_tests / n_test

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_prop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        eps = eta/len(mini_batch)
        self.weights = [w - nw * eps for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - nb * eps for b, nb in zip(self.biases, nabla_b)]

    def back_prop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        acts = self.feed_forward(x)

        delta = self.outp_function(self, acts, y) * acts * (1 - acts)
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(self.activations[-2].T)

        for layer in range(2, self.num_layers):
            delta = self.weights[1-layer].T.dot(delta) * self.activations[-layer] * (1 - self.activations[-layer])
            nabla_b[-layer] = delta
            nabla_w[-layer] = delta.dot(self.activations[-1-layer].T)
        return nabla_b, nabla_w

    def feed_forward(self, acts):

        layer = 1
        self.activations[0] = acts

        for b, w in zip(self.biases, self.weights):
            acts = FC.sigmoid(np.dot(w, acts) + b)
            self.activations[layer] = acts
            layer += 1
        return acts

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


class RegularizedNetwork(Network):
    def __init__(self, sizes, output=True, l1=0, l2=0):
        super().__init__(sizes, output)
        self.l1 = l1
        self.l2 = l2

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        eps = eta / len(mini_batch)
        self.weights = [w - eps * nw - self.l1 * np.sign(w) - self.l2 * w for w, nw in zip(self.weights, nabla_w)]
