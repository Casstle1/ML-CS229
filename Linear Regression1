import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('ex1data1.txt',header=None,names=['Population','profits'])
plt.figure(figsize=(10,10))
plt.scatter(df['Population'],df['profits'],marker='x')
plt.xlabel('Population')
plt.ylabel('Profits')
m=len(df)
X=np.append(np.ones((m,1)), np.array(df["Population"]).reshape(m,1), axis=1)
Y=np.array(df["profits"]).reshape(m,1)
#假设函数为thetaTx直线方程
def h(theta,x):
    return np.transpose(theta)@x
def cost_function(theta,X,Y):
    m=len(X)
    y=X@theta
    return (1/(2*m))*np.sum((y-Y)**2)
initial_theta=np.zeros((2,1))
print(cost_function(initial_theta,X,Y))
def gradient_function(theta,X,Y,alpha,iterations):
    m=len(X)
    J_history=[]
    for i in range(iterations):
        error=X@theta-Y
        grad=(X.T@error)/m
        new_theta=theta-alpha*grad
        J_history.append(cost_function(new_theta,X,Y))
    return new_theta,J_history
alpha=0.01
iterations=1500
theta,J_history=gradient_function(initial_theta,X,Y,alpha,iterations)
print(theta)
plt.plot(X[:,1],Y,'rx',label='训练数据')
plt.plot(X[:,1],X.dot(theta),label='线性回归')
prediction=np.array([1,3.5]).dot(theta)*10000
print(prediction)




