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