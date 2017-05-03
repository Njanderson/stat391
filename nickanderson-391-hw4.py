# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 4

from LinReg import *
import numpy as np
import pandas as pd
import math
from sklearn import discriminant_analysis
from sklearn import linear_model
import matplotlib.pyplot as plt

# Weekly dataset
weekly_data_raw = pd.read_csv("Weekly.csv", usecols=range(1, 10))
mapping = {"Up": 1, "Down": 0}
weekly_data = weekly_data_raw.replace({"Direction" : mapping})
X = weekly_data.ix[:,:-1]
y = weekly_data.ix[:,-1]

# Problem 1

# Reproducibility
random_seed = 1
np.random.seed(random_seed)

def get_k_fold_ind(n, k, prop):
    if len(prop)  != k - 1:
        raise ValueError("Passed k and len(prop) disagree. len(prop) must equal k - 1.")
    all_shuffled = np.random.permutation(n)
    # Appending 1 will include the rest of the elements
    ret = []
    for p in np.append(prop, 1):
        num_ele = math.floor(p * n)
        ret.append(all_shuffled[:num_ele])
        all_shuffled = np.delete(all_shuffled, np.arange(num_ele))

    # Could also use np.split...
    return ret


def model_fit_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# train_test_split splits predictors X and corresponding responses y
# into a test and a training set for X and y, returning
# X_train, X_test, y_train, y_test
# Inspired by sklearn's model_selection's train_test_split
def train_test_split(X, y, test_size):
    n = len(X)
    k = 2
    prop = [test_size]
    idx = get_k_fold_ind(n, k, prop)
    train_idx = idx[0]
    test_idx = idx[1]
    return X.ix[train_idx], X.ix[test_idx], y.ix[train_idx], y.ix[test_idx]

def test_models(X, y, repeat_x):
    scores = pd.DataFrame(columns=['LogReg', 'LDA', 'QDA'])
    for i in range(0, repeat_x):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        model_lr = linear_model.LogisticRegression()
        lr_score = model_fit_score(model_lr, X_train, X_test, y_train, y_test)

        model_lda = discriminant_analysis.LinearDiscriminantAnalysis()
        lda_score = model_fit_score(model_lda, X_train, X_test, y_train, y_test)

        model_qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
        qda_score = model_fit_score(model_qda, X_train, X_test, y_train, y_test)

        i_test_run = pd.DataFrame([[lr_score, lda_score, qda_score]], columns=['LogReg', 'LDA', 'QDA'])
        scores = scores.append(i_test_run, ignore_index=True)
    return scores

# Problem 1a
scores = pd.DataFrame(columns=['LogReg', 'LDA', 'QDA'])
# X.ix[:, 1:3] required due to only using Lag1 and Lag2
X_train, X_test, y_train, y_test = train_test_split(X.ix[:, 1:3], y, test_size=0.5)
model_lr = linear_model.LogisticRegression()
lr_score = model_fit_score(model_lr, X_train, X_test, y_train, y_test)

model_lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda_score = model_fit_score(model_lda, X_train, X_test, y_train, y_test)

model_qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
qda_score = model_fit_score(model_qda, X_train, X_test, y_train, y_test)

i_test_run = pd.DataFrame([[lr_score, lda_score, qda_score]], columns=['LogReg', 'LDA', 'QDA'])
scores = scores.append(i_test_run, ignore_index=True)
print(scores)

repeat_x = 3

# Problem 1b
print("\nTesting models with Lag1 and Lag2")
print(test_models(X.ix[:, 1:3], y, repeat_x))

# Problem 1c
print("\nTesting models with Lag1, Lag2, and Lag3")
print(test_models(X.ix[:, 1:4], y, repeat_x))


# Problem 2

# Fixing random state for reproducibility
x = np.random.randn(100)
eps = np.random.randn(100)
y = x - 2*x**2 + eps

# plt.scatter(x, y)
# plt.show()

x = np.random.randn(100)
eps = np.random.randn(100)
y = x - 2*x**2 + eps

x = np.reshape(x,(x.shape[0], 1))
lin_score = 0
quad_score = 0
cubic_score = 0
quartic_score = 0

def get_model_err(x_train, x_test, y_train, y_test):
    std = standardize(x_train)
    print("Shapes:")
    print(x_train.shape)
    print(std[1].shape)
    print(std[2].shape)
    coef = compute_lsq_estimates(x_train, y_train)
    pred = predict_lsq(x_test, std[1], std[2], coef)
    return (y_test - pred)**2

for i in range(0, len(x)):

    # Delete the ith observations, which will form our x_train and y_train
    x_train = np.delete(x, i)
    y_train = np.delete(x, i)

    # Get our test point
    x_test = np.array(x[i])
    y_test = y[i]

    x_train = np.reshape(x_train,(x_train.shape[0], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 1))
    print(x_train.shape)
    print(x_test.shape)

    # Linear model
    lin_score = lin_score + get_model_err(x_train, x_test, y_train, y_test)

    # Quadratic model
    x_test = np.append(x_test, x_test**2, 1)
    quad = np.append(x_train, x_train**2, 1)
    quad_score = quad_score + get_model_err(quad, x_test, y_train, y_test)

    # Cubic model
    x_test = np.append(x_test, x_test**3, 1)
    cubic = np.append(quad, x_train**3, 1)
    cubic_score = cubic_score + get_model_err(cubic, x_test, y_train, y_test)

    # Quartic model
    x_test = np.append(x_test, x_test**4, 1)
    quartic = np.append(cubic, x_train**4, 1)
    quartic_score = quartic_score + get_model_err(x_train, x_test, y_train, y_test)

print((lin_score, quad_score, cubic_score, quartic_score))