"""providing gradient_descent"""
import os
import sys

from numerical_diff import gradient

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def gradient_descent(f, init_x, lr= 0.01, step_num = 100):
    """경사하강법"""
    x = init_x

    for _ in range(step_num):
        grad = gradient(f, x)
        x -= lr* grad
        print(x)
    return x
