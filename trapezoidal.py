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

print("enter matrix A with each entry in new line")
input_mat(A)
print("enter matrix B with each entry in new line")
input_mat(B)
print("enter matrix u with each entry in new line")
input_mat(u)


print(A)
print(B)
print(u)

def trapezoidal(fun,dt,t0,y0):
    y0=np.reshape(y0,(m,1))
    f0=fun(t0,y0)
    C = y0 + dt/2*f0
    # print("C")
    # print(C)
    I = np.eye(A.shape[0])
    lhs_matrix = I-dt/2*A
    rhs_vector = C + (dt/2) * np.dot(B,u)
    yout = solve(lhs_matrix, rhs_vector)
    # print("yout")
    # print(yout)
    yout=np.reshape(yout,(m))
    return yout        

def system(t,y):
    #y is a 2D array
    y=np.reshape(y,(m,1))
    dy = np.dot(A,y) + np.dot(B,u)
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

plt.subplot(2,1,1)
plt.plot(n,Y[0,:], 'b')
plt.xlabel('Time')
plt.ylabel('Current')
plt.title('Current Vs Time')

plt.subplot(2,1,2)
plt.plot(n, Y[1,:], 'r')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Voltage Vs Time')
print(Y)
plt.show()
