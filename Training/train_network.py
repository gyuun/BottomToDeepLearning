"""Training network with SGD"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Training.two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize= True, one_hot_label= True)
    """ x_train (60000, 784), t_train (60000, 10)
    x_test (10000, 784) t_test (10000, 10)
    """
    train_loss_list = []

    repetition = 10000
    train_size = x_train.shape[0]
    batch_size = 100

    learning_rate = 0.1
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size = 10)

    for i in range(repetition):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.numerical_gradient(x_batch, t_batch) # x_batch (100, 784) t_batch (100, 10)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)

    train_loss_list.append(loss)
