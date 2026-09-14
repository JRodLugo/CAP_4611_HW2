#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    #"""YOUR CODE HERE FOR Q1. Also modify knn.py to implement KNN predict."""
    #raise NotImplementedError()
    for k in [1,3,10]:
        print("k = ", k)
        model = KNN(k)

        #training the model, make predictions
        model.fit(X, y)
        y_predictions_training = model.predict(X)
        y_predictions_test = model.predict(X_test)

        #prints results
        #np.mean is a boolen array, auto checks when prediction was right or wrong
        print("train data accuracy = ", np.mean(y_predictions_training == y))
        print("train err data = ", np.mean(y_predictions_training != y))
        print("test data accuracy = ", np.mean(y_predictions_test == y_test))
        print("test err data = ", np.mean(y_predictions_test != y_test))
        print()

    #generates the plot for k = 1
    modl2 = KNN(1)
    utils.plot_classifier(modl2.fit(X,y), X, y)
    plt.show()


@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    #"""YOUR CODE HERE FOR Q2"""
    ks = list(range(1, 30, 4))
    cv_accs = [] #stores the mean accurancy across folds for each k
    cv_err = [] #stores the mean error across folds for each k 
    num_train_ex = X.shape[0]
    fold_size = num_train_ex // 10
    
    #raise NotImplementedError()
    
    #implement 10-fold cross-validation
    for depth in ks:
        #stores the current fold
        fold_accs = []
        fold_err = []
        #preforms the 10-fold cross validation
        for fold in range(10):
            #begin with everything marked as true
            mask = np.ones(num_train_ex, dtype=bool)
            start = fold * fold_size
            end = (fold  + 1 ) * fold_size

            #mark validation examples as false
            mask[start:end] = False

            #train KNN
            model = KNN(depth)
            model.fit(X[mask], y[mask])

            #make predictions, get the accuracy
            y_predictions = model.predict(X[~mask])
            accuracy = np.mean(y_predictions == y[~mask])
            foldError = np.mean(y_predictions != y[~mask])
            #inputs current accuracy into the array
            fold_accs.append(accuracy)
            fold_err.append(foldError)
        #input folds array into the cross validation array 
        cv_accs.append(np.mean(fold_accs))
        cv_err.append(np.mean(fold_err))

    #test accuracy
    test_accs = []
    test_err = []
    for k in ks:
         #train KNN
        model = KNN(k)
        model.fit(X, y)
        
        #make predictions, get the accuracy
        y_predictions = model.predict(X_test)
        accuracy = np.mean(y_predictions == y_test)
        foldError = np.mean(y_predictions != y_test)

        #store accuracy and error
        test_accs.append(accuracy)
        test_err.append(foldError)

    print("Current ks: ", ks)
    print("Cross-Validation Accuracy: ", cv_accs)
    print("Cross-Validation Error: ", cv_err)
    print("Test Accuracy: ", test_accs)
    print("Test Error: ", test_err)


@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool) #binary matrix of 1s and 0s
    y = dataset["y"]
    #word lists and newsgroup labels for additional newsgroup posts
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    #names of the four newsgroups
    groupnames = dataset["groupnames"]
    #set of words that correspond to each column
    wordlist = dataset["wordlist"]

    # """YOUR CODE HERE FOR Q3.2"""
    # raise NotImplementedError()

    



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    """CODE FOR Q3.4: Modify naive_bayes.py/NaiveBayesLaplace"""

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4. Also modify naive_bayes.py/NaiveBayesLaplace"""
    raise NotImplementedError()



@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE FOR Q4. Also modify random_tree.py/RandomForest"""
    raise NotImplementedError()



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.1. Also modify kmeans.py/Kmeans"""
    raise NotImplementedError()



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.2"""
    raise NotImplementedError()



if __name__ == "__main__":
    main()
