from sklearn import linear_model
import numpy as np
import itertools


def best_subset(X, y):
    lin_reg = linear_model.LinearRegression()
    p_count = X.shape[1]
    # This generates all subsets of the input list
    subset_lengths_to_eval = [list(itertools.combinations(range(0, p_count), length)) for length in range(0, p_count + 1)]
    print(subset_lengths_to_eval)

    var_in_model = np.zeros((p_count, p_count))
    rss_model = []
    # Let's skip the first subset, since we know it's just the average y
    for i, subset_of_length in enumerate(subset_lengths_to_eval[1:]):
        best_rss = -1
        for subset in subset_of_length:
            lin_reg.fit(X[:, subset], y)
            # print("Fitting: " + str(subset))
            rss = np.sum((lin_reg.predict(X[:, subset]) - y)**2)
            if best_rss == -1 or rss < best_rss:
                best_rss = rss
                # Add one since we don't calculate the 0th row
                var_in_model[i] = 0
                var_in_model[i, subset] = 1
        rss_model = rss_model + [best_rss]
    return var_in_model, rss_model

def forward_stepwise(X, y):
    lin_reg = linear_model.LinearRegression()
    p_count = X.shape[1]
    # This generates all subsets of the input list
    subset_lengths_to_eval = [list(itertools.combinations(range(0, p_count), length)) for length in
                              range(0, p_count + 1)]
    print(subset_lengths_to_eval)

    var_in_model = np.zeros((p_count, p_count))
    curr_vars = []
    rss_model = []
    # Let's skip the first subset, since we know it's just the average y
    for i, subset_of_length in enumerate(subset_lengths_to_eval[1:]):
        best_rss = -1
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

# def backward_stepwise(X, y):

# def compute_Cp_bic_adjR2(y, d, rss, var_err):



# Testing on the Credit Data set
import csv
import matplotlib.pyplot as plt
with open('Credit.csv', 'r') as f:
    reader = csv.reader(f)
    credit = np.array(list(reader))[1:, 1:]

X = credit[:, :-1]
y = credit[:, -1:]

# Convert back into list for nice + concat syntax
X_list = [list(x) for x in X]
X = [x[:-4] + [int(x[-4] == 'Male')]  + [int(x[-3] == 'Yes')] + [int(x[-2] == 'Yes')] + [int(x[-1] == 'Asian')] + [int(x[-1] == 'African American')] for x in X_list]
X = [[float(x) for x in x_list] for x_list in X]
y = [[float(y_converted) for y_converted in y_list] for y_list in y]

# Try the various models
# which_p, rss_per_model = best_subset(np.array(X), np.array(y))
which_p, rss_per_model = forward_stepwise(np.array(X), np.array(y))

print(best_subset(np.array(X), np.array(y)))
print(forward_stepwise(np.array(X), np.array(y)))


plt.scatter(range(0, len(rss_per_model)), rss_per_model)
plt.show()
