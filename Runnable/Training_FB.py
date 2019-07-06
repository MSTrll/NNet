import Networks.NN as Net
import Networks.Functions as FC
import numpy as np
import random

import matplotlib.pyplot as plt

data = np.loadtxt("..\Data\data.csv", delimiter=",")

means = data.mean(axis=0)
means[-1] = 0  # правильные ответы мы нормализовывать не будем: это качественные переменные
stds = data.std(axis=0)
stds[-1] = 1
data = (data - means) / stds

np.random.seed(42)
test_index = np.random.choice([True, False], len(data), replace=True, p=[0.25, 0.75])
test = data[test_index]
train = data[np.logical_not(test_index)]

# eye - чтобы создать вертикальный вектор, аналогичный тому, который будет выдавать нейросеть на выходе
train = [(d[:3][:, np.newaxis], np.eye(3, 1, k=-int(d[-1]))) for d in train]
test = [(d[:3][:, np.newaxis], d[-1]) for d in test]

input_count = 3
hidden_count = 6
output_count = 3

random.seed(1)
np.random.seed(1)
nn = Net.Network([input_count, hidden_count, output_count])
nn.SGD(training_data=train, epochs=10, mini_batch_size=10, eta=1, test_data=test)

from ipywidgets import *


@interact(layer1=IntSlider(min=0, max=10, continuous_update=False, description="1st inner layer: ", value=6),
          layer2=IntSlider(min=0, max=10, continuous_update=False, description="2nd inner layer:"),
          layer3=IntSlider(min=0, max=10, continuous_update=False, description="3rd inner layer: "),
          batch_size=BoundedIntText(min=1, max=len(data), value=10, description="Batch size: "),
          learning_rate=Dropdown(options=["0.01", "0.05", "0.1", "0.5", "1", "5", "10"],
                                 description="Learning rate: ")
          )
def learning_curve_by_network_structure(layer1, layer2, layer3, batch_size, learning_rate):
    layers = [x for x in [input_count, layer1, layer2, layer3, output_count] if x > 0]
    nn = Net.Network(int(layers), output=False)
    learning_rate = float(learning_rate)

    CER = []
    cost_train = []
    cost_test = []
    for _ in range(150):
        nn.SGD(training_data=train, epochs=1, mini_batch_size=batch_size, eta=learning_rate)
        CER.append(1 - nn.evaluate(test) / len(test))
        cost_test.append(FC.cost_function(nn, test, onehot=False))
        cost_train.append(FC.cost_function(nn, train, onehot=True))

    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 2, 1)
    plt.ylim(0, 1)
    plt.plot(CER)
    plt.title("Classification error rate")
    plt.ylabel("Percent of incorrectly identified observations")
    plt.xlabel("Epoch number")

    fig.add_subplot(1, 2, 2)
    plt.plot(cost_train, label="Training error", color="orange")
    plt.plot(cost_test, label="Test error", color="blue")
    plt.title("Learning curve")
    plt.ylabel("Cost function")
    plt.xlabel("Epoch number")
    plt.legend()
    plt.show()
