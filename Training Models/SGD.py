import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100,1)), X] # Add x0 = 1 to each instance 

n_epochs = 50
t0, t1, = 5, 50
eta = 0.1
n_iterations = 1000
m = 100
X_new = np.array([[0], [2]])

def learning_scedule(t):
    return t0 / (t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        eta = learning_scedule(epoch * m +i)
        theta = theta - eta * gradients

sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0= 0.1)
sgd_reg.fit(X, y.ravel())

print(theta)
print(sgd_reg.intercept_, sgd_reg.coef_)

