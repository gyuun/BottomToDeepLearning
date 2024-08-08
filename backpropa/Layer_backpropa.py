"""Providing TwoLayerNet with backpropagation"""
import os
import sys

import numpy as np
from collections import OrderedDict

from function_layer import ReLu, Affine, SoftmaxWithLoss

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class TwoLayerNet:
    """2층 신경망"""
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std = 0.01):
        """params는 수치미분, layersms 역전파에 이용"""
        self.params = {}
        self.params['W1'] = weight_init_std \
                            * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std \
                            * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLu()
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """순전파"""
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
        
    def loss(self, x, t):
        """오차"""
        y = self.predict(x)

        return self.lastLayer.forward(y,t)
        
    def accuracy(self, x, t):
        """정확도"""
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy
        
    def gradient(self, x, t):
        """그래디언트"""
        self.loss(x,t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads