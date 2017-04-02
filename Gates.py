import numpy as np


def sigmoid(x, deriv=True):
    if deriv == True:
        return x * (1-x)
    return 1/(1+np.exp(-x))            # 1 / (1 + e^-x)

x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],

])
