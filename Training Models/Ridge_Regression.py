from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
m = 100
X = 6* np.random.rand(m, 1) -3
y = 0.5 * X**2 + X + 2 + np.random.randn(m ,1)


X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2 ,1)), X_new]

ridge_reg = Ridge(alpha =1, solver="cholesky")
ridge_reg.fit(X, y)
print(ridge_reg.predict([[1.5]]))

plt.plot(X_new, ridge_reg.predict([[1.5]]), 'r-')
plt.plot(X, y, 'b.')
plt.show()