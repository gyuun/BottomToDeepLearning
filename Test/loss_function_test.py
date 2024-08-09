"""Testing loss_function"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Training import loss_function

def test_square():
    """오차제곱합 테스트"""
    t = np.array([0,0,1,0,0,0,0,0,0,0])
    y = np.array([0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0])
    assert loss_function.sum_square_error(y, t)


def test_LossFunction():
    """교차 엔트로피 합 테스트"""
    t = np.array([0,0,1,0,0,0,0,0,0,0])
    y = np.array([0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0])
    assert loss_function.cross_entropy_error(y, t)
