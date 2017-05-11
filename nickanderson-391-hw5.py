# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 5

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import numpy as np
import itertools


# best_subset performs Best Subset Selection, considering
# all models from 1-p predictors, returning a 0/1 array
# denoting which predictors are present in each of the models
# from 1 predictor to p predictors that minimize RSS. Also
# returns the RSS for each of the 1-p models.
def best_subset(X, y):
    lin_reg = linear_model.LinearRegression()
    p_count = X.shape[1]

    # This generates all subsets of the input list
    subset_lengths_to_eval = [list(itertools.combinations(range(0, p_count), length)) for length in range(0, p_count + 1)]
    var_in_model = np.zeros((p_count, p_count))
    rss_model = []

    # Let's skip the first subset, since we know it's just the average y
    # due to being the null model
    for i, subset_of_length in enumerate(subset_lengths_to_eval[1:]):
        best_rss = -1
        for subset in subset_of_length:
            lin_reg.fit(X[:, subset], y)
            rss = np.sum((lin_reg.predict(X[:, subset]) - y)**2)

            # Record only the best RSS value seen so far, and record the predictors used
            if best_rss == -1 or rss < best_rss:
                best_rss = rss
                # Add one since we don't calculate the 0th row
                var_in_model[i] = 0
                var_in_model[i, subset] = 1
        rss_model = rss_model + [best_rss]
    return var_in_model, rss_model

# forward_stepwise performs Forward Stepwise Selection, considering
# a subset of the models from 1-p predictors, returning a 0/1 array
# denoting which predictors are present in each of the models
# from 1 predictor to p predictors that minimize RSS. Also
# returns the RSS for each of the 1-p models.
def forward_stepwise(X, y):
    lin_reg = linear_model.LinearRegression()
    p_count = X.shape[1]
    # This generates all subsets of the input list
    subset_lengths_to_eval = [list(itertools.combinations(range(0, p_count), length)) for length in
                              range(0, p_count + 1)]
    var_in_model = np.zeros((p_count, p_count))
    curr_vars = []
    rss_model = []

    # Let's skip the first subset, since we know it's just the average y
    for i, subset_of_length in enumerate(subset_lengths_to_eval[1:]):
        best_rss = -1
        print(subset_of_length)
        subset_with_curr_vars = [subset for subset in subset_of_length if set(curr_vars).issubset(subset)]


        for subset in subset_with_curr_vars:
            lin_reg.fit(X[:, subset], y)
            rss = np.sum((lin_reg.predict(X[:, subset]) - y) ** 2)
            if best_rss == -1 or rss < best_rss:
                best_rss = rss
                # Add one since we don't calculate the 0th row
                var_in_model[i] = 0
                var_in_model[i, subset] = 1
                curr_vars = subset
        rss_model = rss_model + [best_rss]
    return var_in_model, rss_model


# backward_stepwise performs Backward Stepwise Selection, considering
# a subset of the models from 1-p predictors, returning a 0/1 array
# denoting which predictors are present in each of the models
# from 1 predictor to p predictors that minimize RSS. Also
# returns the RSS for each of the 1-p models.
def backward_stepwise(X, y):
    lin_reg = linear_model.LinearRegression()
    p_count = X.shape[1]
    # This generates all subsets of the input list
    subset_lengths_to_eval = [list(itertools.combinations(range(0, p_count), length)) for length in
                              range(0, p_count + 1)]
    var_in_model = np.zeros((p_count, p_count))
    print(var_in_model)
    # Start with all vars
    curr_vars = np.arange(p_count + 1)
    rss_model = []
    # Let's skip the first subset, since we know it's just the average y
    for i, subset_of_length in enumerate(np.flipud(subset_lengths_to_eval[1:])):
        best_rss = -1
        subset_with_curr_vars = [subset for subset in subset_of_length if len(set(curr_vars).intersection(subset)) == len(curr_vars) - 1]
        for subset in subset_with_curr_vars:
            lin_reg.fit(X[:, subset], y)
            rss = np.sum((lin_reg.predict(X[:, subset]) - y) ** 2)
            if best_rss == -1 or rss < best_rss:
                best_rss = rss
                # Add from back to front. Because we remove the first index of subset_lengths,
                # and arrays are 0-indexed, we must subtract 2
                var_in_model[i] = 0
                var_in_model[i, subset] = 1
                curr_vars = subset
        rss_model = [best_rss] + rss_model
    return np.flipud(var_in_model), rss_model

# compute_Cp_bic_adjR2 takes y, a numpy array of responses,
# d, the number of predictors in the model, the RSS of the model,
# and a the variance of the error term in the model, and returns 3
# measures of model quality: Cp, BIC, and Adjusted R^2 in a tuple.
def compute_Cp_bic_adjR2(y, d, rss, var_err):
    n = len(y)
    cp = (1/n) * (rss + 2 * d * (var_err**2))
    bic = (1/(n*(var_err**2))) * (rss + np.log(n) * d * (var_err**2))
    y_bar = np.mean(y)
    tss = np.sum((y - y_bar)**2)
    adj_r_sq = 1 - (rss / (n - d - 1)) * (1/ tss)
    return (cp, bic, adj_r_sq)


# k_fold_cross_validation performs k-fold cross validation, considering
# all models from 1-p predictors, returning a 0/1 array
# denoting which predictors are present in each of the models
# from 1 predictor to p predictors that minimize MSE. Also
# returns the MSE for each of the 1-p models.
# Due to computational limits, also takes largest_considered_subset,
# which bounds the size of subsets considered in k-fold CV.
def k_fold_cross_validation(X, y, k, largest_considered_subset):
    p_count = X.shape[1]

    # This generates all subsets of the input list
    subset_lengths_to_eval = [list(itertools.combinations(range(0, p_count), length)) for length in range(0, p_count + 1)]
    var_in_model = np.zeros((p_count, p_count))
    mse_model = []

    # Let's skip the first subset, since we know it's just the average y
    # We also need to bound by the largest_considered_subset + 1, since slices are exclusive
    for i, subset_of_length in enumerate(subset_lengths_to_eval[1:largest_considered_subset + 1]):
        best_mean_mse = -1
        for subset in subset_of_length:
            # Get the mean of the k-fold CV MSE
            curr_mean_mse = np.mean(cross_val_score(linear_model.LinearRegression(), X[:, subset], y, scoring=make_scorer(mean_squared_error), cv=k))
            if best_mean_mse == -1 or curr_mean_mse < best_mean_mse:
                best_mean_mse = curr_mean_mse
                # Add one since we don't calculate the 0th row
                var_in_model[i] = 0
                var_in_model[i, subset] = 1
        mse_model = mse_model + [best_mean_mse]
    return var_in_model, mse_model

# Problem 1a
n = 100
np.random.seed(0)
X = np.random.randn(n)
eps = np.random.randn(n)

# Problem 1b
b_0 = 2
b_1 = 4
b_2 = 1
b_3 = -2
y = b_0 + b_1 * X + b_2 * X**2 + b_3 * X**3 + eps
X = np.reshape(X, (X.shape[0], 1))
for i in range(2, 11):
    X = np.append(X, np.reshape(X[:,0]**i, (X.shape[0], 1)), 1)

# Problem 1c and 1d
# Calculates quality of fit measures using cp_bic_adjR2 on BSS, FoSS, and BaSS.
# Plots their results

simulated_best_subset_which_p, simulated_best_subset_rss_per_model = best_subset(np.array(X), np.array(y))
simulated_forward_stepwise_which_p, simulated_forward_stepwise_rss_per_model = forward_stepwise(np.array(X), np.array(y))
simulated_backward_stepwise_which_p, simulated_backward_stepwise_rss_per_model = backward_stepwise(np.array(X), np.array(y))

rss_per_method = np.array([simulated_best_subset_rss_per_model, simulated_forward_stepwise_rss_per_model, simulated_backward_stepwise_rss_per_model])

# Creates a 3 x 10 x 3 Array
# The first 3 represents the 3 methods, which are in order: Best Subset Selection, Forward Stepwise Selection, and Backward Stepwise Selection
# 10 represents the 10 different models generated different numbers of predictors from 1-10.
cp_bic_adjR2 = np.array([[compute_Cp_bic_adjR2(y, p, rss, rss / (n - p - 1)) for p, rss in enumerate(rss_per_model)] for rss_per_model in rss_per_method])

fig = plt.figure()
ax1 = fig.add_subplot(111)

# Uncomment any of the three blocks to see its related chart

# plt.title("Cp Scores")
# ax1.scatter(np.arange(1, 11), cp_bic_adjR2[0,:,0], s=50, c='b', marker="s", label='Best Subset Selection')
# ax1.scatter(np.arange(1, 11), cp_bic_adjR2[1,:,0], s=50, c='r', marker="o", label='Forward Stepwise Selection')
# ax1.scatter(np.arange(1, 11), cp_bic_adjR2[2,:,0], s=50, c='g', marker="x", label='Backward Stepwise Selection')
# plt.legend(loc='upper right')

# plt.title("BIC Scores")
# ax1.scatter(np.arange(1, 11), cp_bic_adjR2[0,:,1], s=50, c='b', marker="s", label='Best Subset Selection')
# ax1.scatter(np.arange(1, 11), cp_bic_adjR2[1,:,1], s=50, c='r', marker="o", label='Forward Stepwise Selection')
# ax1.scatter(np.arange(1, 11), cp_bic_adjR2[2,:,1], s=50, c='g', marker="x", label='Backward Stepwise Selection')
# plt.legend(loc='lower right')

# plt.title("AdjR2 Scores")
# ax1.scatter(np.arange(1, 11), cp_bic_adjR2[0,:,2], s=50, c='b', marker="s", label='Best Subset Selection')
# ax1.scatter(np.arange(1, 11), cp_bic_adjR2[1,:,2], s=50, c='r', marker="o", label='Forward Stepwise Selection')
# ax1.scatter(np.arange(1, 11), cp_bic_adjR2[2,:,2], s=50, c='g', marker="x", label='Backward Stepwise Selection')
# plt.legend(loc='upper right')

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


# Problem 2a: Forward Stepwise Selection

college_forward_stepwise_which_p, college_forward_stepwise_rss_per_model = forward_stepwise(np.array(X), np.array(y))

# Problem 2b: Backward Stepwise Selection
college_backward_stepwise_which_p, college_backward_stepwise_rss_per_model = backward_stepwise(np.array(X), np.array(y))


rss_per_method = np.array([college_forward_stepwise_rss_per_model, college_backward_stepwise_rss_per_model])
cp_bic_adjR2 = np.array([[compute_Cp_bic_adjR2(y, p, rss, rss / (n - p - 1)) for p, rss in enumerate(model)] for model in rss_per_method])


# Uncomment any of the two blocks to see its related chart

# plt.title("AdjR2 Scores")
# ax1.scatter(np.arange(1, len(college_forward_stepwise_rss_per_model)+1), cp_bic_adjR2[0,:,2], s=50, c='r', marker="o", label='Forward Stepwise Selection')
# ax1.scatter(np.arange(1, len(college_backward_stepwise_rss_per_model)+1), cp_bic_adjR2[1,:,2], s=50, c='g', marker="x", label='Backward Stepwise Selection')

# plt.title("Cp Scores")
# ax1.scatter(np.arange(1, len(college_forward_stepwise_rss_per_model)+1), cp_bic_adjR2[0,:,0], s=50, c='r', marker="o", label='Forward Stepwise Selection')
# ax1.scatter(np.arange(1, len(college_backward_stepwise_rss_per_model)+1), cp_bic_adjR2[1,:,0], s=50, c='g', marker="x", label='Backward Stepwise Selection')

# plt.title("BIC Scores")
# ax1.scatter(np.arange(1, len(college_forward_stepwise_rss_per_model)+1), cp_bic_adjR2[0,:,1], s=50, c='r', marker="o", label='Forward Stepwise Selection')
# ax1.scatter(np.arange(1, len(college_backward_stepwise_rss_per_model)+1), cp_bic_adjR2[1,:,1], s=50, c='g', marker="x", label='Backward Stepwise Selection')

# plt.legend(loc='upper right')
# plt.show()

# Problem 2a: 10-fold CV

# max_pred_count = 5
# college_cv_which_p, college_cv_mse_per_model = k_fold_cross_validation(np.array(X), y, 10, max_pred_count)

# plt.title("MSE Scores")
# ax1.scatter(np.arange(1, max_pred_count+1), college_cv_mse_per_model, s=50, c='r', marker="o", label='10-Fold Cross Validation')
#
# plt.show()

# --------------------------TESTING--------------------------

# I included the tests that I performed on the Credit Data Set

# import csv
# with open('Credit.csv', 'r') as f:
#     reader = csv.reader(f)
#     credit = np.array(list(reader))[1:, 1:]
#
# X = credit[:, :-1]
# y = credit[:, -1:]

# Convert back into list for nice + concat syntax
# X_list = [list(x) for x in X]
# X = [x[:-4] + [int(x[-4] == 'Male')]  + [int(x[-3] == 'Yes')] + [int(x[-2] == 'Yes')] + [int(x[-1] == 'Asian')] + [int(x[-1] == 'African American')] for x in X_list]
# X = [[float(x) for x in x_list] for x_list in X]
# y = [[float(y_converted) for y_converted in y_list] for y_list in y]

# Try the various models
# which_p, rss_per_model = best_subset(np.array(X), np.array(y))
# which_p, rss_per_model = forward_stepwise(np.array(X), np.array(y))
# which_p, rss_per_model = backward_stepwise(np.array(X), np.array(y))
# print(best_subset(np.array(X), np.array(y)))
# print(forward_stepwise(np.array(X), np.array(y)))
# print(backward_stepwise(np.array(X), np.array(y)))

# plt.scatter(range(0, len(rss_per_model)), rss_per_model)
# plt.show()
