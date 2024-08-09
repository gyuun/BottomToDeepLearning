"""describing backpropagation layers"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Training.loss_function import cross_entropy_error
from Network.network import softmax

class MulLayer:
    """곱셉 노드 클래스
    
    forward : 순전파 메소드
    인스턴스 변수 초기화

    backward : 역전파 메소드
    return : xy를 바꿔서 입력값에 곱한 값
    """
    def __init__(self):
        """곱셉노드의 순전파 변수 선언"""
        self.x = None
        self.y = None

    def forward(self, x, y):
        """순전파 변수에 출력값 저장"""
        self.x = x
        self.y = y

        return x*y

    def backward(self, dout):
        """역전파 수행"""
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    """덧셈 노드 클래스"""
    def __init__(self):
        """역전파는 출력값을 그대로 반환하므로 순전파의 출력 저장 x"""
        pass

    def forward(self, x, y):
        """바로 출력하는 모습"""
        return x+y

    def backward(self, dout):
        """미분값이 x+y 함수의 편미분이 1임을 나타내는 * 1 연산"""
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class ReLu:
    """ReLu함수 노드"""
    def __init__(self):
        """리스트의 mask 기능을 이용"""
        self.mask = None

    def forward(self, x):
        """ReLu 노드의 순전파"""
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0 # true 즉 0보다 작으면 0이다

        return out

    def backward(self, dout):
        """ReLu 노드의 역전파"""
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    """Sigmoid 노드"""
    def __init__(self):
        """순전파의 출력값 저장할 변수 선언"""
        self.out = None

    def forward(self, x):
        """Sigmoid 노드의 순전파. 출력값을 저장한다."""
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        """Sigmoid 노드의 역넌파. 순전파의 출력을 이용한다."""
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    """Affine 노드"""
    def __init__(self, W, b):
        """가중치와 편향 값을 클래스에 저장"""
        self.X = None
        self.W = W
        self.b = b
        self.dW = None
        self.db = None

    def forward(self, X):
        """Affine 노드의 순전파. 행렬곱 연산한다"""
        self.X = X
        out = np.dot(X, self.W)

        return out

    def backward(self, dout):
        """Affine 노드의 역전파. 전치행렬을 이용한다."""
        dX = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis = 0)

        return dX


class SoftmaxWithLoss:
    """출력층 노드"""
    def __init__(self):
        """출력층 노드의 생성자"""
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        """출력층 노드의 순전파"""
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout = 1):
        """출력층 노드의 역전파"""
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
