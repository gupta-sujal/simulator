import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve

#n= input("Enter the number of states: ");

#i = input("Enter thr number of inputs:");

A = np.array([ [-1, -1], [1,0] ]);
B = np.array([ [1],[0] ]);

Vs = float(input("Enter the input supply voltage:"));
print(A);
print(B);

def trap(fun, dt, t0, y0):
    f0 = fun(t0,y0);
    C = y0 + dt/2*f0;
    I = np.eye(A.shape[0]);
    lhs_matrix = I-dt/2*A;
    rhs_vector = C + (dt/2) * B.flatten()*Vs;
    yout = solve(lhs_matrix, rhs_vector);
    return yout

def system(t,y):
    #y is a 2D array
    dy = A.dot(y) + B.flatten()*Vs;
    return np.array(dy);

y0 = np.array( [ 0, 0] );
dt = 0.01;
T = 10;
N = int(T/dt);
n = np.linspace(0,T,N);

Y = np.zeros((2,N));
Y[:,0] = y0;
yin = y0;
for i in range(N-1):
    yout = trap(system, dt, n[i], yin);
    Y[:,i+1] = yout;
    yin = yout;

plt.figure()

plt.subplot(2,1,1);
plt.plot(n,Y[0,:], 'b');
plt.xlabel('Time');
plt.ylabel('Current');
plt.title('Current Vs Time');

plt.subplot(2,1,2);
plt.plot(n, Y[1,:], 'r');
plt.xlabel('Time');
plt.ylabel('Voltage');
plt.title('Voltage Vs Time');
print(Y);
plt.show();