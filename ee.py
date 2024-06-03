import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#n= input("Enter the number of states: ");

#i = input("Enter thr number of inputs:");

A = np.array([ [-1, -1], [1,0] ]);
B = np.array([ [1],[0] ]);

Vs = float(input("Enter the input supply voltage:"));
print(A);
print(B);


def rk4(fun, dt, t0,y0):
    f1 = fun(t0,y0);
    f2 = fun(t0 + dt/2 , y0 + (dt/2)*f1);
    f3 = fun(t0 + dt/2 , y0 + (dt/2)*f2);
    f4 = fun(t0 + dt , y0 + dt*f3);
    yout = y0 + (dt/6)*(f1 + 2*f2 + 2*f3 + f4);
    return yout;

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
    yout = rk4(system, dt, n[i], yin);
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
