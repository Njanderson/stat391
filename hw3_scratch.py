# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 3
import numpy as np

# Utility and Unstandardized versions of functions

# compute_lsq_estimates_unstandardized returns the least squares coefficient estimates
# for a linear regression using the predictors in design_mat with responses y.
# Returns UNSTANDARDIZED coefficients that be used directly for predictions
def compute_lsq_estimates_unstandardized(design_mat, y):
    # β = (X^T X)^(−1) X^T y
    # First standardize the predictors, adding a 1 column for the y-intercept, Beta_0
    stdized_w_ones = standardize_design(design_mat)
    tposed = np.transpose(stdized_w_ones)
    inv_squared = np.linalg.inv(np.dot(tposed, stdized_w_ones))
    stdized_coeff = np.ndarray.flatten(np.dot(np.dot(inv_squared, tposed), y))
    # stdized_coeff now contains a flattened array of coefficients, but Beta_0 needs to be fit.
    # Use y_bar = b_hat_0 +  b_hat_1 * x_bar_1 + ...
    # b_hat_0 = y_bar - b_hat_1 * x_bar - .....
    stdized = standardize(design_mat)
    x_bar = stdized[1]
    x_std_dev = stdized[2]
    stdized_coeff[0] = stdized_coeff[0] - np.sum(x_bar * np.delete(stdized_coeff, 0, 0) / x_std_dev)
    # Rescale beta_hat_i, i > 0
    stdized_coeff[1:] = stdized_coeff[1:] / x_std_dev
    return stdized_coeff

# compute_prediction returns predictions with a multiple linear regression model
# using the unstandardized, final coefficients and an array of predictors
def compute_prediction(predictors, model):
    return [model[0] + np.sum(obs*np.delete(model, 0, 0)) for obs in predictors]

# heart_data = np.loadtxt("hw3_data.csv", delimiter = "," ,skiprows =1)
# phones = heart_data[:,0] # number of phones per 1000 inhabitants
# saturated = heart_data[:,1] # proportion of saturated fat
# animal = heart_data[:,2] # proportion of animal fat
# deaths = heart_data[:,3] # deathrate due to heart disease

# Question 1:
# print(standardize_design(heart_data[:,0:3]))


# print(heart_data[:,0:3].shape)
# Question 2:
# lsr = compute_lsq_estimates(phones, deaths)
# means_std = standardize(phones)
# predictions = [predict_lsq(obs, means_std[1], means_std[2], lsr) for obs in phones]
# print(compute_F_stat_lsq(deaths, predictions, 1))

# With 20 degrees of freedom with a p value of 0.05: 1.717
# Our value is 5.6616981568, which is more extreme, so we reject the null hypothesis

# Question 3:
# lsr = compute_lsq_estimates(heart_data[:,0:2], deaths)
# print(lsr)

# Question 4:
# lsr = compute_lsq_estimates(heart_data[:,0:2], deaths)
# means_std = standardize(heart_data[:,0:2])
# predictions = [predict_lsq(obs, means_std[1], means_std[2], lsr) for obs in heart_data[:,0:2]]
# f_stat = compute_F_stat_lsq(deaths, predictions, 2)

# Question 5:
# compute_R2_lsq
# b
# lsr = compute_lsq_estimates(phones, deaths)
# means_std = standardize(phones)
# predictions = [predict_lsq(obs, means_std[1], means_std[2], lsr) for obs in phones]
# print(compute_R2_lsq(deaths, predictions))
# print(compute_std_err_lsq(phones, deaths, predictions)[0])

# c
# lsr = compute_lsq_estimates(heart_data[:,0:2], deaths)
# means_std = standardize(heart_data[:,0:2])
# predictions = [predict_lsq(obs, means_std[1], means_std[2], lsr) for obs in heart_data[:,0:2]]
# print(compute_R2_lsq(deaths, predictions))
# print(compute_std_err_lsq(heart_data[:,0:2], deaths, predictions)[0])


# R2 hardly changes between b and c, so it's not a good addition to add saturated, same with RSE

# Question 6:
# lsr = compute_lsq_estimates(heart_data[:,0:3], deaths)
# print(lsr)
# 108 phones per 1000 inhabitants, 33%
# of saturated fat for men between the ages of 55 and 59, 7% of animal fat
# for men between the ages of 55 and 59.
# means_std = standardize(heart_data[:,0:3])
# print(predict_lsq([[108, 33, 7]], means_std[1], means_std[2], lsr))
#
# lsr = compute_lsq_estimates_unstandardized(heart_data[:,0:3], deaths)
# print(compute_prediction([[108, 33, 7]], lsr))

# Question 7:
# saturated = heart_data[:,1] # proportion of saturated fat
# animal = heart_data[:,2] # proportion of animal fat
# sat_ani_interaction = np.reshape(saturated * animal, (saturated.shape[0], 1))
# fats = np.append(heart_data[:,1:3], sat_ani_interaction, 1)
# lsr = compute_lsq_estimates(fats, deaths)
# print(lsr)
# means_std = standardize(fats)
# predictions = [predict_lsq(obs, means_std[1], means_std[2], lsr) for obs in fats]
# print(compute_F_stat_lsq(deaths, predictions, 3))

# my_data = np.genfromtxt('Advertising.csv', delimiter=',')
# labels_removed = np.delete(np.delete(my_data, 0, axis=0), 0, axis=1)
# pred = labels_removed[:,0:3]
# resp = np.ndarray.flatten(labels_removed[:,3])
# lsr = compute_lsq_estimates(pred, resp)
# lsr_unstd = compute_lsq_estimates_unstandardized(pred, resp)
# print(lsr)

# def predict_lsq(X_test, bar_X, std_X, hat_beta):

# means_std = standardize(pred)
#
# predictions = [predict_lsq(obs, means_std[1], means_std[2], lsr) for obs in pred]
# print(compute_std_err_lsq(pred, np.ndarray.flatten(resp), predictions))
# print(compute_R2_lsq(resp, predictions))
# print(compute_F_stat_lsq(resp, predictions, 3))

#
# lsr = compute_lsq_estimates_unstandardized(pred, resp)
# print(compute_std_err_lsq(pred, np.ndarray.flatten(resp), compute_prediction(pred, lsr)))
# print(compute_R2_lsq(resp, compute_prediction(pred, lsr)))
# print(compute_F_stat_lsq(resp, compute_prediction(pred, lsr), 3))
#
#
#
# Call:
# lm(formula = Sales ~ TV + Radio + Newspaper, data = auto)
#
# Residuals:
#     Min      1Q  Median      3Q     Max
# -8.8277 -0.8908  0.2418  1.1893  2.8292
#
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)
# (Intercept)  2.938889   0.311908   9.422   <2e-16 ***
# TV           0.045765   0.001395  32.809   <2e-16 ***
# Radio        0.188530   0.008611  21.893   <2e-16 ***
# Newspaper   -0.001037   0.005871  -0.177     0.86
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 1.686 on 196 degrees of freedom
# Multiple R-squared:  0.8972,	Adjusted R-squared:  0.8956
# F-statistic: 570.3 on 3 and 196 DF,  p-value: < 2.2e-16
