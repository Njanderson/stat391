# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 4

import numpy as np
import pandas as pd
from sklearn import discriminant_analysis
from sklearn.model_selection import train_test_split
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

def model_fit_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

def test_models(X, y, repeat_x):
    scores = pd.DataFrame(columns=['LogReg', 'LDA', 'QDA'])
    for i in range(0, repeat_x):
        # Split into test and train: only use Lag1 and Lag2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
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
np.random.seed(1)
x = np.random.randn(100)
eps = np.random.randn(100)
y = x - 2*x**2 + eps
