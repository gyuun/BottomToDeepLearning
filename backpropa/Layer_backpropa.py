import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from collections import OrderedDict
from backpropa.function_layer import ReLu, Affine, SoftmaxWithLoss
from Training.numerical_diff import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std = 0.01):
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
            for layer in self.layers.values():
                x = layer.forward(x)

            return x
        
        def loss(self, x, t):
            y = self.predict(x)

            return self.lastLayer.forward(y,t)
        
        def accuracy(self, x, t):
            y = self.predict(x)
            y = np.argmax(y, axis = 1)
            if t.ndim != 1:
                t = np.argmax(t, axis = 1)
            accuracy = np.sum(y == t) / float(x.shape[0])

            return accuracy
        
        def numerical_gradient(self, x, t):
            loss_W = lambda W: self.loss(x,t)

            grads = {}
            grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
            grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
            grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
            grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

            return grads
        
        def gradient(self, x, t):
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