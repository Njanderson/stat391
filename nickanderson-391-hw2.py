# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 2

import numpy as np
import matplotlib.pyplot as plt
import unittest

# Assigned Functions

# Calculates the mean of a numpy array
def my_mean(x):
    x = np.float64(x)
    bar_x = np.sum(x) / np.size(x)
    return bar_x

# Calculates the covariance between x and y, numpy arrays of size n
# Throws ValueError if x and y don't share the same size
def my_cov(x, y):
    if (np.size(x) != np.size(y)):
        raise ValueError('Provided arrays have different dimensions!')
    n = np.size(x)
    x_bar = my_mean(x)
    y_bar = my_mean(y)
    x_sub = x - x_bar
    y_sub = y - y_bar
    return (1/n) * np.sum(x_sub * y_sub)

# Computes hat_beta_0 and hat_beta_1 for linear regression
# Returns an array with the index i containing hat_beta_i
def compute_simple_lsq_estimates(x, y):
    if (np.size(x) != np.size(y)):
        raise ValueError('Provided arrays have different dimensions!')
    # Makes sure we have float64s for precision even if we weren't passed arrays of them
    x = np.float64(x)
    y = np.float64(y)
    xy_bar = my_mean(x * y)
    x_bar = my_mean(x)
    y_bar = my_mean(y)
    x_sq_bar = my_mean(x*x)
    x_bar_sq = x_bar*x_bar
    hat_beta_1 = (xy_bar - x_bar * y_bar) / (x_sq_bar - x_bar_sq)
    hat_beta_0 = y_bar - hat_beta_1*x_bar
    return [hat_beta_0, hat_beta_1]

# Displays a plot of input data along with the data's least squares regression
# line. Plots the least squares regression line between
# the maximum and minimum x values in the input data.
def plot_simple_lsq(x, y, label_pred, label_resp):
    coeff = compute_simple_lsq_estimates(x, y)
    plt.plot(x, y, 'o')
    ind_pts_fit_line = np.array([np.amin(x), np.amax(x)])
    dep_pts_fit_line = coeff[1] * ind_pts_fit_line + coeff[0]
    plt.plot(ind_pts_fit_line, dep_pts_fit_line)
    plt.xlabel(label_pred)
    plt.ylabel(label_resp)
    plt.show()

# Testing and Utility Functions

# Generate covariance matrix similar to numpy's cov function
def my_cov_matrix(x, y):
    return [[my_cov(x,x),my_cov(x,y)], [my_cov(y,x),my_cov(y,y)]]

# For testing, generates a toy covariant dataset
# If first roll is divisible by two, use that as the second roll value. Otherwise, roll again for the second roll.
def gen_covariant_rolls():
    first_roll = np.random.randint(1, 6+1, 1000)
    second_roll = [np.random.randint(1, 6+1) if x%2 != 0 else x for x in first_roll]
    return [first_roll, second_roll]

# For testing, generates linear data with some error term
# Takes coefficients, normal error term's standard deviation, and size
def gen_linear_data_with_error(beta_0, beta_1, err_std_dev, size, min, max):
    err = np.random.normal(0, err_std_dev, size)
    ind_pts = np.random.randint(min, max, size)
    return [ind_pts, beta_1 * ind_pts + beta_0 + err]

# Testing

class TestLinearRegression(unittest.TestCase):
    min_bound = -10000
    max_bound = 10000
    lin_reg_size = 10000

    def test_mean(self):
        bound = np.random.randint(1, TestLinearRegression.max_bound + 1)
        data = np.arange(0, bound)
        self.assertEqual(my_mean(data), np.average(data))

    # Tests the my_cov function against numpy's implementation
    def test_cov(self):
        cov_rolls = gen_covariant_rolls()
        # Flatten and compare within some lambda
        mine = my_cov_matrix(cov_rolls[0], cov_rolls[1])
        true_cov = np.cov(cov_rolls[0], cov_rolls[1])
        self.assertAlmostEqual(mine[0][0], true_cov[0][0], 2)
        self.assertAlmostEqual(mine[0][1], true_cov[0][1], 2)
        self.assertAlmostEqual(mine[1][0], true_cov[1][0], 2)
        self.assertAlmostEqual(mine[1][1], true_cov[1][1], 2)


    # Tests the linear regression function, compute_simple_lsq_estimates, against numpy's implementation
    def test_lin_reg(self):
        # Add 1 because np.random.randin is exclusive on the upper bound
        min_bound = TestLinearRegression.min_bound
        max_bound = TestLinearRegression.max_bound + 1
        beta_0 = np.random.randint(min_bound, max_bound)
        beta_1 = np.random.randint(min_bound, max_bound)
        std_dev = np.random.randint(1, max_bound)
        min = np.random.randint(min_bound, 1)
        max = np.random.randint(0, max_bound)
        test_data = gen_linear_data_with_error(beta_0, beta_1, std_dev, TestLinearRegression.lin_reg_size, min, max)
        mine = compute_simple_lsq_estimates(test_data[0], test_data[1])
        true_val = np.polyfit(test_data[0], test_data[1], 1)
        self.assertAlmostEqual(mine[0], true_val[1], 5)
        self.assertAlmostEqual(mine[1], true_val[0], 5)

if __name__ == '__main__':
    unittest.main()
