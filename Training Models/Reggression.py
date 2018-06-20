import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


X = 6* np.random.rand(m, 1) -3
y = 0.5 * X**2 + X + 2 + np.random.randn(m ,1)

X_b = np.c_[np.ones((100,1)), X] # Add x0 = 1 to each instance 
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2 ,1)), X_new] # Add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)

#Sklearn version

lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)

plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()


