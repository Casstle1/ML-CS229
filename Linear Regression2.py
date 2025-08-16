import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("ex1data2.txt",header=None,names=["Size","Bedrooms","Price"])
m=len(data)
x0=np.ones(m)
size=np.array(data["Size"])
bedrooms=np.array(data["Bedrooms"])
X=np.array([x0,size,bedrooms]).T
y=np.array(data["Price"]).reshape(-1,1)
theta_init=np.zeros((3,1))
raw_X=X.copy()
def normalize(X):
    mu=np.mean(X[:,1:],axis=0)
    sigma=np.std(X[:,1:],axis=0)
    X_norm=X.copy()
    X_norm[:,1:]=(X[:,1:]-mu)/sigma
    return X_norm,mu,sigma
X,mu,sigma=normalize(X)
def cost_function(theta,X,Y):
    m=len(X)
    y=X@theta
    return (1/(2*m))*np.sum((y-Y)**2)
def gradient_descent(X,y,theta,alpha,iterations):
    J_history=[]
    for i in range(iterations):
        error=X@theta-y
        grad=(X.T@error)/m
        theta=theta-alpha*grad
        J_history.append(cost_function(theta,X,y))
    return theta,J_history
alpha=0.01
iterations=1500
theta,J_history=gradient_descent(X,y,theta_init,alpha,iterations)
plt.plot(J_history)
plt.title("Jdecrease")
plt.xlabel("iterations")
plt.ylabel("J")
price=theta.T@np.array([1,
                        (1650-mu[0])/sigma[0],
                        (3-mu[1])/sigma[1]])
print(price)
theta=np.linalg.inv(raw_X.T@raw_X)@raw_X.T@y
price1=theta.T@np.array([1,1650,3])
print(price1)
