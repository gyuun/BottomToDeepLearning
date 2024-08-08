"""Providing actvating function"""
import numpy as np
def stair(x):
    if x<=0 :
        return 0
    else :
        return 1
def sigmoid(x) :
    e = np.exp(1)
    return 1/(1 + e**(-x))
def ReLu(x) :
    return np.maximum(0,x)
def identity(x):
    return x
def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c)
    sum = np.sum(exp_a)
    return exp_a/sum
