#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import utils

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
    modl2.fit(X, y)
    utils.plot_classifier(modl2, X, y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('../figs/q1_knn_boundary.png', dpi=180, bbox_inches='tight')
    plt.close()


@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    cv_accs = [] #stores the mean accurancy across folds for each k
    cv_err = [] #stores the mean error across folds for each k 
    num_train_ex = X.shape[0]
    fold_size = num_train_ex // 10
    
    
    #implement 10-fold cross-validation
    for depth in ks:
        #stores the current fold
        fold_accs = []
        fold_err = []
        #preforms the 10-fold cross validation
        for fold in range(10):
            #begin with everything marked as true
            mask = np.ones(num_train_ex, dtype=bool)
            start = fold * num_train_ex // 10
            end = (fold + 1) * num_train_ex // 10

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

    train_errors = []
    for k in ks:
        model = KNN(k)
        model.fit(X, y)
        train_errors.append(float(np.mean(model.predict(X) != y)))
    print('Training errors:', train_errors)
    plt.figure()
    plt.plot(ks, cv_accs, 'o-', label='10-fold cross-validation')
    plt.plot(ks, test_accs, 's-', label='Test')
    plt.xlabel('Number of neighbours, k')
    plt.ylabel('Accuracy')
    plt.xticks(ks)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../figs/q2_cv_test_accuracy.png', dpi=180)
    plt.close()
    plt.figure()
    plt.plot(ks, train_errors, 'o-')
    plt.xlabel('Number of neighbours, k')
    plt.ylabel('Training error (fraction misclassified)')
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig('../figs/q2_training_error.png', dpi=180)
    plt.close()


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

    print('Word 73:', wordlist[72])
    print('Words in example 803:', np.asarray(wordlist)[X[802]])
    print('Group:', groupnames[y[802]])

    



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

    for beta in [0, 1, 10000]:
        model = NaiveBayes(4) if beta == 0 else NaiveBayesLaplace(4, beta)
        model.fit(X, y)
        probabilities = model.p_xy[:, 0]
        print('beta:', beta, 'class 0 probabilities:', probabilities)
        print('Zeros:', np.sum(probabilities == 0),
              'Range:', probabilities.min(), probabilities.max())
        print('Training error:', np.mean(model.predict(X) != y))
        print('Validation error:', np.mean(model.predict(X_valid) != y_valid))



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

    np.random.seed(4611)
    print('Random tree (seed 4611)')
    evaluate_model(RandomTree(max_depth=np.inf))
    np.random.seed(4611)
    print('Random forest (50 trees, seed 4611)')
    evaluate_model(RandomForest(num_trees=50, max_depth=np.inf))



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

    np.random.seed(4611)
    best_error = np.inf
    for _ in range(50):
        model = Kmeans(k=4)
        model.fit(X)
        error = model.error(X, model.predict(X), model.means)
        if error < best_error:
            best_error, best_model = error, model
    print('Lowest error:', best_error)
    print('Error history of best run:', best_model.errors_)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=best_model.predict(X), cmap='tab10', s=15)
    plt.scatter(best_model.means[:, 0], best_model.means[:, 1], c='black', marker='X', s=110, label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Best of 50 initializations: k = 4, SSE = {best_error:.6f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../figs/q5_best_clustering.png', dpi=180)
    plt.close()



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    np.random.seed(4611)
    errors = []
    for k in range(1, 11):
        best_error = np.inf
        for _ in range(50):
            model = Kmeans(k)
            model.fit(X)
            best_error = min(best_error, model.error(X, model.predict(X), model.means))
        errors.append(best_error)
    print('Minimum errors for k=1,...,10:', errors)
    plt.figure()
    plt.plot(range(1, 11), errors, 'o-')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Minimum sum of squared distances')
    plt.xticks(range(1, 11))
    plt.title('Best of 50 initializations per k')
    plt.tight_layout()
    plt.savefig('../figs/q5_elbow.png', dpi=180)
    plt.close()



if __name__ == "__main__":
    main()
