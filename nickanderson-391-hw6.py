# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 6
import SubsetSelection
import numpy as np
from sklearn.utils import shuffle

n = 100
np.random.seed(0)
X = np.reshape(np.random.randn(n), (n, 1))
eps = np.reshape(np.random.randn(n), (n, 1))

b_0 = 2
b_2 = 1
b_7 = 4
b_12 = -2
b_17 = -2

y = b_0 + b_2 * X**2 + b_7 * X**7 + b_12 * X**12 + b_17 * X**17 + eps

X, y = shuffle(X, y, random_state=0)

# 2 - 21 due to starting with adding quadratic,
# then the end is exclusive.
# for i in range(2, 21):


X = np.array([X**i for i in range(1, 21)])
X = np.reshape(X, (X.shape[0], X.shape[1]))
X = X.transpose()
print(X.shape)
print(y.shape)
train_prop = 0.1

X_train = X[0:int(X.shape[0] * train_prop)]
y_train = y[0:int(y.shape[0] * train_prop)]

X_test = X[int(X.shape[0] * train_prop):]
y_test = y[int(y.shape[0] * train_prop):]

var_in_model, rss_model = SubsetSelection.best_subset(X_train, y_train)

# We want to evaluate each of var_in_model, using each of these
# to calculate MSE.

print(var_in_model)
print(rss_model)
# X_per_model = X[var_in_model]
# print(X_per_model.shape)
# mse_per_model