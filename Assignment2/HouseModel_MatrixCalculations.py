#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 03:54:41 2020

@author: riku
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv

#path to file, contains both Y and X values
filepath = 'Assignment2_PyInputs.csv'
#read in file
input_file = pd.read_csv(filepath)
#pull out X matrix
X = input_file.loc[:, 'X0':'X4'].to_numpy()
#pull out normalized Y matrix
Y_n = input_file.Y_Actual.to_numpy()
#pull out Y matrix
Y = input_file.Y_Actual.to_numpy()
# Golden Rule:
# B_hat = (X_tpose * X)_inv * X_tpose * Y

X_tpose = X.transpose()

B_hat = inv(X_tpose.dot(X)).dot(X_tpose).dot(Y) # uses the non-normalized Y
#B_hat_n = inv(X_tpose.dot(X)).dot(X_tpose).dot(Y) # this also works, but then Y_hat is also normalized
print("B_hat:")
print(B_hat)
np.savetxt("output/B_hat.csv", B_hat, delimiter=",")

Y_hat = X.dot(B_hat)

print("Y_hat:")
print(Y_hat) # this matches the excel model prices exactly, no need to denormalize
np.savetxt("output/Y_hat.csv", Y_hat, delimiter=",")

Y = input_file.Y_Actual.to_numpy()

error = Y - Y_hat
print("error:")
print(error) # this matches the excel error values exactly
np.savetxt("output/error.csv", error, delimiter=",")

#sum of errors
SSE = np.sum(error * error)
#number of samples
n = len(Y)
#number of factors
k = 4

sig_hat_squared = SSE/(n - (k+1))
print("variance of errors:")
print(sig_hat_squared)

var_covar_matrix = sig_hat_squared * inv(X_tpose.dot(X))

print("variance-covariance matrix:")
print(var_covar_matrix)
np.savetxt("output/var_covar_matrix.csv", var_covar_matrix, delimiter=",")

std_error_B_hat = (np.diag(var_covar_matrix))**(1/2)

t_stat = B_hat/std_error_B_hat

print("t-stat:")
print(t_stat) #this matches the t-stat from excel lin reg exactly
np.savetxt("output/t_stat.csv", t_stat, delimiter=",")

#if t-stat is within the 95% t-distribution then we reject that factor