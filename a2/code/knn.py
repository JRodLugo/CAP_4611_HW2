"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        #"""YOUR CODE HERE FOR Q1"""
        #fills in the k-nearest neigbor prediction rule
        #distance = num of training points(i) * num of test points(j)
        distance = euclidean_dist_squared(self.X, X_hat)

        #array to hold the predictions for each test case
        Y_hat = np.zeros(X_hat.shape[0])

        #goes through the test points
        for i in range(X_hat.shape[0]):
            #gets and sorts the current distance
            cur_dist = distance[:, i]
            sorted_indices = np.argsort(cur_dist)
            #Gets first k indicies, gets label of k nearest training points, Predict most common label
            Y_hat[i] = utils.mode(self.y[sorted_indices[:self.k]])

        #return the predictions
        return Y_hat 
        #raise NotImplementedError()


