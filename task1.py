#  through this task we aim at solving differential eqautions in matrix form using RK4 method with some initial conditions
# 

import numpy as np

def sample_diff_eq(x,params):
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    delta = params["delta"]
    # np.reshape(x,(2,1))
    # print(x)
    A=np.matrix([[alpha,-1*beta*x[0]],[delta*x[1],-1*gamma]])
    xdot=np.array([alpha*x[0]-1*beta*x[0]*x[1],delta*x[1]*x[0]-1*gamma*x[1]])
    return xdot


def RK4(f,x0,t0,tf,dt):
    # tf is final time
    # t0 is initial start time
    # x0 are the inital conditions
    # dt is the time step
    # f is the function in consideration

    t=np.arange(t0,tf,dt)
    n=t.size #the no of time points
    nv=x0.size #the no of state variables

    x=np.zeros(shape=(nv,n))

    x[:,0]=x0
    for i in range(n-1):
        k1=f(t[i],x[:,i])
        k2=f(t[i]+dt/2,x[:,i]+dt*k1/2)
        k3=f(t[i]+dt/2,x[:,i]+dt*k2/2)
        k4=f(t[i]+dt,x[:,i]+dt*k3)

        x[:,i+1]=x[:,i]+(dt/6)*(k1+2*k2+2*k3+k4)

    return x[:,n-1]


x0=np.array([5,10])
t0=1
tf=4
dt=1e-3
params={"alpha":1,"beta":2,"gamma":3,"delta":4}
f=lambda t,x:sample_diff_eq(x,params)

x=RK4(f,x0,t0,tf,dt)
print(x)


    