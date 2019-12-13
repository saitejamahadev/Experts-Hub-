class LogisticRegression:
    def __init__(self,lr=0.01,num_iter=10000):
        self.lr=lr
        self.num_iter = num_iter
        
    def _sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def _loss(self,h,y):
        return (-y*np.log(h)-(1-y)*np.log(1-h)).mean()
    
    def fit(self,x,y):
        self.theta=np.zeros(x.shape[1])
        for i in range(self.num_iter):
            z=np.dot(x,self.theta)
            h=self._sigmoid(z)
            gradient= np.dot(x.T,(h-y))/y.size
            self.theta -= self.lr*gradient
        
    def predict_prob(self,x):
        return self._sigmoid(np.dot(x,self.theta))
    
    def predict(self,x,threshold):
        return self.predict_prob(x) >= threshold
 
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1
model = LogisticRegression(lr=0.1, num_iter=300000)
model.fit(X, y)
preds = model.predict(X,0.5)
(preds == y).mean()
model.theta

#sklearn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1e20)
model.fit(X,y)
preds = model.predict(X)
(preds == y).mean()
model.intercept_, model.coef_

