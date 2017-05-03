# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 4

import numpy as np
import pandas as pd
import math
from sklearn import discriminant_analysis
# from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Weekly dataset
weekly_data_raw = pd.read_csv("Weekly.csv", usecols=range(1, 10))
mapping = {"Up": 1, "Down": 0}
weekly_data = weekly_data_raw.replace({"Direction" : mapping})
X = weekly_data.ix[:,:-1]
y = weekly_data.ix[:,-1]

# Problem 1

# Reproducibility
random_seed = 0
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

# Implementation of sklearn's model_selection's train_test_split
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
        # Split into test and train: only use Lag1 and Lag2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        # print(X_train)
        # print(y_train)
        model_lr = linear_model.LogisticRegression()
        lr_score = model_fit_score(model_lr, X_train, X_test, y_train, y_test)

        model_lda = discriminant_analysis.LinearDiscriminantAnalysis()
        lda_score = model_fit_score(model_lda, X_train, X_test, y_train, y_test)

        model_qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
        qda_score = model_fit_score(model_qda, X_train, X_test, y_train, y_test)

        i_test_run = pd.DataFrame([[lr_score, lda_score, qda_score]], columns=['LogReg', 'LDA', 'QDA'])
        scores = scores.append(i_test_run, ignore_index=True)
    return scores


repeat_x = 3
print("\nTesting models with Lag1 and Lag2")
print(test_models(X.ix[:, 1:3], y, repeat_x))
print("\nTesting models with Lag1, Lag2, and Lag3")
print(test_models(X.ix[:, 1:4], y, repeat_x))


# Problem 2

# Fixing random state for reproducibility
# np.random.seed(0)
# x = np.random.randn(100)
# eps = np.random.randn(100)
# y = x - 2*x**2 + eps
