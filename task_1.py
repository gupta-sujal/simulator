import numpy as np
from matplotlib import pyplot as plt

def rungeKutta(f, x0, y0, h):
    k1 = h * f(x0, y0)
    k2 = h * f(x0 + 0.5 * h, y0 + 0.5 * k1)
    k3 = h * f(x0 + 0.5 * h, y0 + 0.5 * k2)
    k4 = h * f(x0 + h, y0 + k3)
  
    y_out = y0 + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_out

def f(x, y):
    return x + y*2

X = 5
h = 0.1
num_points = int(X / h)
y0 = [2, 5, 6]
x0 = np.linspace(0, X, num_points)

Y = np.zeros((3, num_points))
Y[:, 0] = y0

y_in = y0

for i in range(num_points - 1):
    y_out = rungeKutta(f, x0[i], y_in, h)
    Y[:, i] = y_out
    y_in = y_out


print(y_out)
