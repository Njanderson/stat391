# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 3
import numpy as np

# standardize accepts X_std, a numpy array of arrays,
# returning a standardized version of the array, the column means,
# and the column standard deviations
def standardize(X_std):
    col_means = np.mean(X_std, 0)
    std = np.std(X_std, 0)
    stdized = (X_std - col_means) / std
    return (stdized, col_means, std)

def standardize_with_mean_std(X_std, mean, std_dev):
    stdized = (X_std - mean) / std_dev
    return (stdized, mean, std_dev)

# add_ones accepts M, a numpy array of arrays,
# returning a numpy array with a column of 1's pre-appended
def add_ones(M):
    # If a dimension needs to be added...
    if len(M.shape) < 2:
        M = np.reshape(M, (M.shape[0], 1))
    return np.insert(M, 0, 1, axis=1)

# standardize_design accepts design_mat, a numpy array,
# returning a standardized version of the array with a
# column of 1's pre-appended
def standardize_design(design_mat):
    stdized = standardize(design_mat)[0]
    return add_ones(stdized)

# compute_lsq_estimates returns the least squares coefficient estimates
# for a linear regression using the predictors in design_mat with responses y.
# Returns STANDARDIZED coefficients that CANNOT be used directly for predictions with unstandardized data
# Returns flattened coefficients
def compute_lsq_estimates(design_mat, y):
    # β = (X^T X)^(−1) X^T y
    # First standardize the predictors, adding a 1 column for the y-intercept, Beta_0
    stdized_w_ones = standardize_design(design_mat)
    tposed = np.transpose(stdized_w_ones)
    inv_squared = np.linalg.inv(np.dot(tposed, stdized_w_ones))
    stdized_coeff = np.ndarray.flatten(np.dot(np.dot(inv_squared, tposed), y))
    return stdized_coeff

# predict_lsq accepts X_test, an observation for which to find a prediction,
# bar_X, the predictor column means, std_x, the predictor column standard deviations,
# and returns a prediction using the linear regression model described
# by hat_beta, with predictor mean bar_X and standard deviation std_X.
def predict_lsq(X_test, bar_X, std_X, hat_beta):
    # hat_beta now contains a flattened array of coefficients, but Beta_0 needs to be fit.
    # Use y_bar = b_hat_0 +  b_hat_1 * x_bar_1 + ...
    # b_hat_0 = y_bar - b_hat_1 * x_bar - .....
    X_test_stdized = (X_test - bar_X) / std_X
    return np.sum(X_test_stdized * hat_beta[1:]) + hat_beta[0]

# compute_std_err_lsq accepts design_mat, a numpy 2-d array of predictors,
# y, the true value associated with the predictors, and hat_y, the predictions
# for the predictors described by design_mat, returning a tuple, with
# tuple[0] = rse, and tuple[1] = SE(Beta), a SE array
def compute_std_err_lsq(design_mat, y, hat_y):
    # RSE = sqrt((1/n-p-1)*RSS)
    if len(design_mat.shape) < 2:
        design_mat = np.reshape(design_mat, (design_mat.shape[0], 1))
    n, p = float(design_mat.shape[0]), float(design_mat.shape[1])
    rse = np.sqrt((1.0/(n-p-1.0))*np.sum(np.square(y - hat_y)))
    # (X^T X)^(−1)
    stdized_w_ones = standardize_design(design_mat)
    tposed = np.transpose(stdized_w_ones)
    inv_squared = np.linalg.inv(np.dot(tposed, stdized_w_ones))
    return (rse, rse*np.sqrt(np.diagonal(inv_squared)))

# compute_R2_lsq accepts y, the true response values, and hat_y,
# the predicted response values, returning the R^2 statistic
def compute_R2_lsq(y, hat_y):
    # R^2 = 1 - RSS / TSS
    RSS = np.sum(np.square(y - hat_y))
    y_bar = np.mean(y)
    TSS = np.sum(np.square(y - y_bar))
    return 1.0 - RSS / TSS

# compute_F_stat_lsq accepts y, the true response values, and hat_y,
# the predicted response values, and p, the number of predictors,
# returning the F_statistic
def compute_F_stat_lsq(y, hat_y, p):
    # F = ((TSS - RSS) / p) / (RSS / (n - p - 1))
    n = float(y.shape[0])
    RSS = np.sum(np.square(y - hat_y))
    y_bar = np.mean(y)
    TSS = np.sum(np.square(y - y_bar))
    # print("Degrees of freedom for F-stat: " + str(n - p - 1))
    return ((TSS - RSS) / p) / (RSS / (n - p - 1))