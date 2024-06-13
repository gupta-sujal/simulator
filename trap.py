import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve

m = int(input("Enter the number of states: "))

k = int(input("Enter the number of inputs:"))

# A and B are taken for state equation
A = np.zeros((m,m),dtype=float)
B = np.zeros((m,k),dtype=float)
u = np.zeros((k,1),dtype=float)

def input_mat(A):
    m=np.shape(A)[0]
    n=np.shape(A)[1]
    for i in range(m):
        for j in range(n):
            value = float(input(f"Enter value for matrix[{i}][{j}]: "))  # Taking float input for generality
            A[i, j] = value

input_mat(A)
input_mat(B)
input_mat(u)


print(A)
print(B)
print(u)

def trapezoidal(fun,dt,t0,y0):
    f0=fun(t0,y0)
    C = y0 + dt/2*f0
    # print("C")
    # print(C)
    I = np.eye(A.shape[0])
    lhs_matrix = I-dt/2*A
    rhs_vector = C + (dt/2) * np.dot(B,u).flatten()
    yout = solve(lhs_matrix, rhs_vector)
    # print("yout")
    # print(yout)
   # yout=np.reshape(yout,(m))
    return yout        

def system(t,y):
    #y is a states array
    dy = np.dot(A,y) + np.dot(B,u).flatten()
    return dy

y0 = np.zeros(m)
dt = 0.01
T = 10
N = int(T/dt)
n = np.linspace(0,T,N)

Y = np.zeros((m,N))
Y[:,0] = y0
yin = y0
for i in range(N-1):
    yout = trapezoidal(system, dt, n[i], yin)
    Y[:,i+1] = yout
    yin = yout

plt.figure()

for i in range(m):
    plt.subplot(m, 1, i + 1)
    plt.plot(n, Y[i, :], label=f'State {i + 1}')
    plt.xlabel('Time')
    plt.ylabel(f'State {i + 1}')
    plt.title(f'State {i + 1} vs Time')
    plt.legend()
print(Y)
plt.show()
