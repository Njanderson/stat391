# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 7

from LinReg import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


from numpy import genfromtxt

# fit_svm_linear fits a linear-kernel SVM to the predictors X,
# using 10-fold CV for estimating test error. Prints out the error
# associated with each value of c and plots the resulting error
# function that's a value of c.
def fit_svm_linear(X, y, c_arr):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    linear_svm = np.array([(c, np.mean(cross_val_score(SVC(kernel='linear', C=c), X, y, cv=10, scoring=make_scorer(mean_squared_error)))) for c in c_arr])
    print("Linear SVM C Value Comparison with 10 Fold CV")

    # Plot Linear SVM
    plt.title("Linear SVM C Value Comparison with 10 Fold CV")
    ax1.scatter(linear_svm[:,0], linear_svm[:, 1])
    print(linear_svm)

    plt.legend(loc='upper right')
    plt.show()

# fit_svm_rbf fits a radial basis function-kernel SVM to the predictors X,
# using 10-fold CV for estimating test error. Prints out the error
# associated with each value of c and plots the resulting error
# function that's a value of c for each gamma in gamma_arr.
def fit_svm_rbf(X, y, gamma_arr, c_arr):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for gamma in gamma_arr:
        rbf_svm = np.array([(c, np.mean(cross_val_score(SVC(kernel='rbf', C=c, gamma=gamma), X, y, cv=10, scoring=make_scorer(mean_squared_error)))) for c in c_arr])
        print(rbf_svm)
        ax1.scatter(rbf_svm[:, 0], rbf_svm[:, 1], label='Gamma=' + str(gamma))
    plt.legend(loc='upper right')
    plt.show()

# fit_svm_poly fits a polynomial-kernel SVM to the predictors X,
# using 10-fold CV for estimating test error. Prints out the error
# associated with each value of c and plots the resulting error
# function that's a value of c for each degree in degree_arr.
def fit_svm_poly(X, y, degree_arr, c_arr):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for degree in degree_arr:
        poly_svm = np.array([(c, np.mean(cross_val_score(SVC(kernel='poly', C=c, degree=degree), X, y, cv=10, scoring=make_scorer(mean_squared_error)))) for c in c_arr])
        print(poly_svm)
        ax1.scatter(poly_svm[:, 0], poly_svm[:, 1], label='Degree=' + str(degree))
    plt.legend(loc='upper right')
    plt.show()

# Auto data set
auto_data = genfromtxt("Auto.csv", delimiter=',', skip_header=1)
X = auto_data[:, 1:-1]
y = np.reshape(auto_data[:, 0], (auto_data.shape[0], 1))

# median_mpg_relation is a  binary variable that takes on a 1
# for cars with gas mileage above the median, and a 0 for cars
# with gas mileage below the median
median_mpg = np.median(y)
median_mpg_relation = np.array([1 if mpg > median_mpg else 0 for mpg in y])

# C values to try for all SVM
c_arr = np.arange(1, 10)
np.random.seed(0)

# standardize for speed
X = standardize(X)[0]

fit_svm_linear(X, median_mpg_relation, c_arr)

# Gamma values to try in Radial Basis Function SVM
gamma_arr = np.arange(1, 10)

fit_svm_rbf(X, median_mpg_relation, gamma_arr, c_arr)

# Orders of Polynomials to try in SVM
degree_arr = np.arange(1, 10)

fit_svm_poly(X, median_mpg_relation, degree_arr, c_arr)