import numpy as np

x=np.linspace(1,100,200)

y=3*x+1


theta=np.random.rand(2,1)


def gradinetdescent(y,theta,x,alpha):
    y_predict=theta[1]*x+theta[0]
    theta[0]=theta[0]-alpha*np.sum(y_predict-y)*1/len(x)
    theta[1]=theta[1]-alpha*np.sum((y_predict-y)*x)*1/len(x)
    return theta

alpha=0.0001
iteration= 10000
for i in range(iteration):
    theta=gradinetdescent(y,theta,x,alpha)
    
y_predict=theta[1]*x+theta[0]   
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(y,x,color='red')
plt.plot(y_predict,x,color='blue')
plt.legend(['actual','predicted'])


from sklearn.linear_model import LinearRegression

lin=LinearRegression()
x=x.reshape(-1,1)
y=y.reshape(-1,1)
lin.fit(x,y)
lin.intercept_
lin.coef_

    
    
