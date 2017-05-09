import numpy as np
import pandas as p

from numpy.linalg import inv

def h_t(z):
    return 1.0/(1 + np.exp(-z))

X = p.read_csv('logistic_x.txt', delim_whitespace=True, dtype=np.float64)
y = p.read_csv('logistic_y.txt', delim_whitespace=True, dtype=np.float64)
