import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#逻辑回归logistic regression
data=pd.read_csv('ex2data1.txt',header=None,names=["Exam 1 score","Exam 2 score","Accepted"])
m=len(data['Accepted'])
x0=np.ones(m)
size=np.array(data['Exam 1 score'])
bedrooms=np.array(data['Exam 2 score'])
X=np.array([x0,size,bedrooms]).T
y=data['Accepted'].values
m,n=X.shape
pos=np.where(y==1)[0]
neg=np.where(y==0)[0]
plt.plot(X[pos,1],X[pos,2],'b+',label='Admitted')
plt.plot(X[neg,1],X[neg,2],'yo',label='Not Admitted')
plt.legend()
def sigmoid(x):
    return 1/(1+np.exp(-x))
def compute_cost(theta,X,y):
    m=len(y)
    h=sigmoid(X@theta)#100,1
    return (-y.T@np.log(h)-(1-y).T@np.log(1-h))/m
def compute_gradient(theta,X,y):
    m=len(y)
    h=sigmoid(X@theta)
    gradient=(X.T@(h-y))/m
    return gradient
initial_theta=np.zeros(X.shape[1])
result=minimize(compute_cost,initial_theta,args=(X,y),method='CG',jac=compute_gradient,options={'maxiter':500,'disp':1})
theta=result.x
print(theta)
plot_x=np.array([min(X[:,1]-2),max(X[:,1]+2)])
plot_y=(-1/theta[2])*(theta[1]*plot_x+theta[0])
plt.plot(plot_x,plot_y,label='Decision Boundary')
plt.legend()
prob=sigmoid(np.array([1,45,85]).dot(theta))#通过概率为0.77
print(prob)
p=(sigmoid(X@theta)>=0.5).astype(int)
print(np.mean(p==y)*100)
data=pd.read_csv('ex2data2.txt',header=None,names=["Text1","Test2","Status"])
X=data[["Text1","Text2"]].values
y=data["Status"].values
print(X)
def map_feature(X1,X2,degree=6):
    X1=np.asarray(X1).reshape(-1,1)
    X2=np.asarray(X2).reshape(-1,1)
    out=np.ones((len(X1),1))
    for i in range(1,degree+1):
        for j in range(i+1):
            out=np.hstack([out,(X1**(i-j))*(X2**j)])
    return out
X=map_feature(X[:,0],X[:,1],degree=6)
def compute_regularized_cost(theta,X,y,_lamda):
    m=len(y)
    reg=_lamda/(2*m)*np.sum(theta[1:]**2)
    return compute_cost(theta,X,y)+reg
def compute_regularized_gradient(theta,X,y,_lamda):
    m=len(y)
    grad=compute_gradient(theta,X,y)
    grad[1:]+=_lamda/m*grad[1:]
    return grad
_lamda=1
initial_theta=np.ones(X.shape[1])
result=minimize(compute_regularized_cost,initial_theta,args=(X,y,_lamda),method='CG',jac=compute_regularized_gradient,options={'maxiter':400,'disp':1})
theta=result.x
u=np.linspace(-1,1.5,50)
v=np.linspace(-1,1.5,50)
z=np.zeros((len(u),len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i,j]=map_feature(u[i],v[j],6).dot(theta)
plt.contour(u,v,z.T,levels=[0])
plt.show()
