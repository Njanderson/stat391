# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 6

import LinReg
import numpy as np
from sklearn.utils import shuffle
from sklearn import linear_model
import matplotlib.pyplot as plt

# Problem 1

# split_test_train takes X, predictions, and y, a response,
# and prop, a proportion to use for testing, and returns
# X_train, y_train, X_test, y_test.
# Note: Pre-shuffle X, y for randomly split test and train sets
def split_test_train(X, y, train_prop):
    X_train = X[0:int(X.shape[0] * train_prop)]
    y_train = y[0:int(y.shape[0] * train_prop)]
    X_test = X[int(X.shape[0] * train_prop):]
    y_test = y[int(y.shape[0] * train_prop):]
    return X_train, y_train, X_test, y_test

# Generating a random 1000-sized data set
n = 1000
np.random.seed(0)
X = np.reshape(np.random.randn(n), (n, 1))
eps = np.reshape(np.random.randn(n), (n, 1))

# Non-zero Coefficients
b_2 = 30
b_7 = -100
b_10 = 50
b_15 = -25
b_19 = -15

# Creating the response
y = b_2 * X**2 + b_7 * X**7 + b_10 * X**10 + b_15 * X**15 + b_19 * X**19 + eps

# Shuffle data for later partitioning into test and train
X, y = shuffle(X, y, random_state=0)

# Add 20 features.
# Use range(1, 21) because range is upper-bound exclusive
X = np.array([X**i for i in range(1, 21)])
X = np.reshape(X, (X.shape[0], X.shape[1]))

# Transpose because current arrangement is reversed,
# so each row currently labels each feature, instead
# of having each row be an observation
X = X.transpose()

# Use 10% of the data as data set as training data
train_prop = 0.1

# Partition data into test and training data
X_train, y_train, X_test, y_test = split_test_train(X, y, train_prop)

# Perform Best Subset Selection
# var_in_model, rss_model = SubsetSelection.best_subset(X_train, y_train)

# Instead of running the preceding best_subset function, due to how long the computation takes,
# I recorded the results in the following array.

# Results of BSS
var_in_model = [[ 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   1,   0],
 [ 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,
   1,   0],
 [ 0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   1,   0,   0,   0,
   1,   0],
 [ 0,   0,   0,   0,   0,   0,   1,   0,   0,   1,   0,   0,   0,   0,   1,   0,   0,   0,
   1,   0],
 [ 0,   1,   0,   0,   0,   0,   1,   0,   0,   1,   0,   0,   0,   0,   1,   0,   0,   0,
   1,   0],
 [ 0,   1,   0,   1,   0,   0,   1,   0,   0,   1,   0,   0,   0,   0,   1,   0,   0,   0,
   1,   0],
 [ 0,   1,   0,   1,   0,   0,   1,   0,   0,   1,   0,   0,   0,   1,   1,   0,   0,   0,
   1,   0],
 [ 0,   1,   0,   1,   0,   0,   1,   0,   1,   1,   0,   0,   0,   0,   1,   1,   0,   0,
   1,   0],
 [ 0,   1,   0,   1,   0,   1,   1,   0,   1,   1,   0,   0,   0,   0,   1,   0,   0,   0,
   1,   1],
 [ 0,   1,   0,   0,   0,   0,   1,   1,   0,   1,   0,   0,   0,   0,   1,   1,   1,   1,
   1,   1],
 [ 0,   1,   0,   0,   0,   0,   1,   0,   0,   1,   0,   1,   1,   0,   1,   1,   1,   1,
   1,   1],
 [ 0,   1,   0,   0,   0,   0,   1,   0,   0,   1,   1,   0,   1,   1,   1,   1,   1,   1,
   1,   1],
 [ 0,   1,   0,   0,   0,   0,   1,   0,   1,   1,   1,   0,   1,   1,   1,   1,   1,   1,
   1,   1],
 [ 0,   1,   0,   0,   0,   0,   1,   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1],
 [ 0,   1,   0,   0,   0,   1,   1,   1,   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1],
 [ 0,   1,   0,   0,   1,   1,   1,   1,   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1],
 [ 0,   1,   1,   1,   1,   0,   1,   1,   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1],
 [ 0,   1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1],
 [ 0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1],
 [ 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   1,   1], ]

# Use as Mask
var_in_model = np.array([[bit == 1 for bit in bit_arr] for bit_arr in var_in_model])

# Collect MSE of each model
mse = []

# Calculate MSE by fitting each model then evaluating
lin_reg = linear_model.LinearRegression()
for var_mask in var_in_model:
    lin_reg.fit(X_train[:, var_mask], y_train)
    mse.append(np.average((lin_reg.predict(X_test[:, var_mask]) - y_test) ** 2))

# plt.title("Log(MSE) Per Model with D Features")
# plt.scatter(range(1, len(mse) + 1), mse)
# plt.show()

RSS = np.array([17282129979856.871, 26477556745.215881, 70711472.053559601, 44620.436417584628,
 75.104742927828752, 72.257454628269414, 68.295460814142587, 68.240687445940068,
 68.229722321860763, 68.004296417428662, 67.801775323241756, 67.046620033167159,
 66.830153565806057, 66.526652571038483, 65.521527374785322, 65.288909107415023,
 65.111192686406213, 64.907357441733069, 64.631152567481948, 64.446085435415924])

mse_train = RSS / n

# plt.title("Log(Training MSE) Per Model with D Features")
# plt.scatter(range(1, len(mse_train) + 1), mse_train)
# plt.show()

true_model = np.zeros(20)
true_model[1] = 30
true_model[6] = -100
true_model[9] = 50
true_model[14] = -25
true_model[18] = -15

root_square_coef_error = []

for model in var_in_model:
    lin_reg.fit(X_train[:, model], y_train)

    placed = 0
    est_coef = np.zeros(20)
    # Make 20 length array with predictors slotted in their model positions
    for i, coef in enumerate(model):
        if coef:
            est_coef[i] = lin_reg.coef_[0, placed]
            placed = placed + 1

    root_square_coef_error.append(np.sum((true_model - est_coef)**2)**0.5)

# plt.title("Coefficient Error Per Model with D Features")
# plt.scatter(range(1, len(root_square_coef_error) + 1), root_square_coef_error)
# plt.show()

# Problem 2

import pandas as pd
# College data set
college_data_raw = pd.read_csv("College.csv")
mapping = {"Yes":1,"No":0}
college_data = college_data_raw.replace({"Private":mapping})
cols = [col for col in college_data.columns if col not in ["Unnamed: 0", "Apps"]]
X = college_data[cols]
y = college_data["Apps"]

np.random.seed(0)

# Use 10% of the data as data set as training data
train_prop = 0.8

# Shuffle data for later partitioning into test and train
X, y = shuffle(X, y, random_state=1)

X_train, y_train, X_test, y_test = split_test_train(X, y, 0.8)

X_train_mean = np.mean(X_train, 0)
X_train_std = np.std(X_train, 0)

# Standardize both the test and the train data using the training set
X_test = np.array(LinReg.standardize_with_mean_std(X_test, X_train_mean, X_train_std)[0])
X_train = np.array(LinReg.standardize(X_train)[0])


lin_reg.fit(X_train, y_train)
test_mse = np.mean((y_test - lin_reg.predict(X_test))**2)
linreg_rss = np.sum((y_test - lin_reg.predict(X_test))**2)


lam_grid = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
k = 10

# RidgeCV Regression
ridge = linear_model.RidgeCV(lam_grid, cv=k)
ridge.fit(X_train, y_train)
ridge_err = np.mean((ridge.predict(X_test) - y_test)**2)
ridge_rss = np.sum((ridge.predict(X_test) - y_test)**2)

# LassoCV Regression
lasso = linear_model.LassoCV(alphas=lam_grid, cv=k)
lasso.fit(X_train, y_train)
lasso_err = np.mean((lasso.predict(X_test) - y_test)**2)
lasso_rss = np.sum((lasso.predict(X_test) - y_test)**2)

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.title("Ridge Regression Predictions")

# Convert to mask array
subset_selection_model = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
subset_selection_model = np.array([bit == 1 for bit in subset_selection_model])

# Fit masked array
lin_reg.fit(X_train[:, subset_selection_model], y_train)
test_mse = np.mean((y_test - lin_reg.predict(X_test[:, subset_selection_model]))**2)
subset_selection_rss = np.sum((y_test - lin_reg.predict(X_test[:, subset_selection_model]))**2)

model_rss = [linreg_rss, ridge_rss, lasso_rss]
adj_r2_per_model = []
y_bar = np.mean(y_test)
tss = np.sum((y_test - y_bar) ** 2)
n = len(y_test)

# For these models, they use all of the predictors
d = len(ridge.coef_)
for rss_for_model in model_rss:
    adj_r_sq = 1 - (rss_for_model/(n-d-1))/(tss/(n-1))
    adj_r2_per_model.append(adj_r_sq)

# Found through calculations in hw5
subset_selection_adjr2 = 0.92326546
adj_r2_per_model.append(subset_selection_adjr2)


