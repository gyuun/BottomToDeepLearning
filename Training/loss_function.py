"""Providing loss function"""
import numpy as np

def sum_square_error (y, t):
    """오차 제곱합"""
    return np.sum((y-t)**2)


def cross_entropy_error (y, t):
    """교차 엔트로피 합
    미니배치의 경우 평균을 이용한다
    미니배치 입력일 경우 : y,t 는 2차원 배열이다."""
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    result = -np.sum(t * np.log(y + delta)) / batch_size
    return result
