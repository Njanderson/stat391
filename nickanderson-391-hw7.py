# Nick Anderson
# 1328354
# nja4@uw.edu
# Stat 391: HW 7

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from numpy import genfromtxt

# Auto data set
auto_data = genfromtxt("Auto.csv", delimiter=',', skip_header=1)
X = auto_data[:, 1:-1]
y = np.reshape(auto_data[:, 0], (auto_data.shape[0], 1))

# median_mpg_relation is a  binary variable that takes on a 1
# for cars with gas mileage above the median, and a 0 for cars
# with gas mileage below the median
median_mpg = np.median(y)
median_mpg_relation = np.array([1 if mpg > median_mpg else 0 for mpg in y])


c_arr = np.arange(1, 10)


np.random.seed(0)

scores = np.array([(c, np.mean(cross_val_score(SVC(kernel='linear', C=c), X, median_mpg_relation, cv=10))) for c in c_arr])
print("Linear")
print(scores)

print("RBF")
gamma_arr = np.arange(1, 10)
for gamma in gamma_arr:
    scores = np.array([(c, np.mean(cross_val_score(SVC(kernel='rbf', C=c, gamma=gamma), X, median_mpg_relation, cv=10))) for c in c_arr])
    print("Gamma=" + str(gamma))
    print(scores)

print("Polynomial")
degree_arr = np.arange(1, 10)
for degree in degree_arr:
    print("Degree=" + str(degree))
    scores = np.array([(c, np.mean(cross_val_score(SVC(kernel='poly', C=c, degree=degree), X, median_mpg_relation, cv=10))) for c in c_arr])
    print(scores)
