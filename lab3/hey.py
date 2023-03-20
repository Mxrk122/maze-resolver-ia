import math
import numpy as np
from matplotlib import pyplot as plt

from gradient_descent import gradient_descent
from linear_cost import linear_cost, linear_cost_gradient


TRAINING_SET_SIZE = 200

x = np.linspace(-10, 30, TRAINING_SET_SIZE)

X = np.vstack(
    (
        np.ones(TRAINING_SET_SIZE),
        x,
        x ** 2,
        x ** 3,
    )
).T

y = (5 + 2 * x ** 3 + np.random.randint(-9000, 9000, TRAINING_SET_SIZE)).reshape(
    TRAINING_SET_SIZE,
    1
)

m, n = X.shape
theta_0 = np.random.rand(n, 1)
r_theta, costs, thetas = gradient_descent(
    X,
    y,
    theta_0,
    linear_cost,
    linear_cost_gradient,
    learning_rate=0.00000001,
    threshold=0.001,
    max_iter=10000
)


for test_theta in thetas:
    pass
plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], X @ test_theta, color='green')
plt.show()

print("test_theta", test_theta)