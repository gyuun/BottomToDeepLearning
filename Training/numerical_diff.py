import numpy as np

def numerical_diff (f, x):
    h = 1e-50
    return f(x + h) - f(x) /h

def central_num_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_2(x):
    """x1^2 + x2^2 를 반환한다.
    
    parameter : ndarray
    return : float
    """
    return x[0]**2 + x[1]**2
def partial_diff(f, x):
    h= 1e-4
    """x1^2 + x2^2 의 편미분을 출력한다. """
    res = []
    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)

        x[i] = tmp - h
        fxh2 = f(x)

        x[i] = tmp
        res.append((fxh1 - fxh2)/ (2*h))
    print('dx0 ', res[0], 'dx1', res[1])


def numerical_gradient(f, x):
    """f에 대해 x 에서의 gradient 를 반환한다.
    
    parameter : (f : 미분할 함수), (x : ndarray)
    return : ndarray
    """
    print('x is ',x.shape)

    h = 1e-4
    grad = np.zeros_like(x)
    if x.ndim == 2: # 가중치
        for i in range(len(x)):
            for idx in range(len(x[0])):
                tmp_val = x[i, idx]
                x[i, idx] = tmp_val + h
                fxh1 = f(x)
                x[i, idx] = tmp_val - h
                fxh2 = f(x)
                grad[i, idx] = (fxh1 - fxh2) / (2*h)
                x[i, idx] = tmp_val
            print(i)
    elif x.ndim == 1: # 편향
        for i in range(x.size):
            tmp_val = x[i]
            x[i] = tmp_val + h
            fxh1 = f(x)
            x[i] = tmp_val - h
            fxh2 = f(x)
            grad[i] = (fxh1 - fxh2) / (2*h)
            x[i] = tmp_val
    return grad








