"""Small deterministic checks for the completed assignment algorithms."""
import unittest
import numpy as np
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest
from kmeans import Kmeans


class AssignmentTests(unittest.TestCase):
    def test_knn_distance_and_vote(self):
        X = np.array([[0.], [2.], [3.]])
        y = np.array([0, 1, 1])
        model = KNN(1)
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        model = KNN(3)
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(np.array([[0.1]])), [1])

    def test_bayes_hand_counts(self):
        X = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0],
                      [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0],
                      [1, 0, 0], [0, 0, 0]])
        y = np.array([0]*6 + [1]*4)
        model = NaiveBayes(2)
        model.fit(X, y)
        np.testing.assert_allclose(model.p_y, [.6, .4])
        np.testing.assert_allclose(model.p_xy, [[1/6, 3/4], [5/6, 1/4], [4/6, 1/4]])
        np.testing.assert_array_equal(model.predict(np.array([[1, 1, 0]])), [1])
        smooth = NaiveBayesLaplace(2, 1)
        smooth.fit(X, y)
        np.testing.assert_allclose(smooth.p_xy, [[2/8, 4/6], [6/8, 2/6], [5/8, 2/6]])
        zero = NaiveBayesLaplace(2, 0)
        zero.fit(X, y)
        np.testing.assert_allclose(zero.p_xy, model.p_xy)

    def test_forest_votes_and_refit(self):
        class FixedTree:
            def __init__(self, labels): self.labels = labels
            def predict(self, X): return np.array(self.labels)
        forest = RandomForest(3, 2)
        forest.trees = [FixedTree([0, 1]), FixedTree([1, 1]), FixedTree([1, 0])]
        np.testing.assert_array_equal(forest.predict(np.zeros((2, 1))), [1, 1])
        X = np.arange(8).reshape(-1, 1)
        y = np.array([0]*4 + [1]*4)
        np.random.seed(7)
        forest.fit(X, y)
        forest.fit(X, y)
        self.assertEqual(len(forest.trees), 3)

    def test_kmeans_sse_and_monotonicity(self):
        model = Kmeans(2)
        self.assertEqual(model.error(np.array([[0., 0.], [3., 4.]]),
                                    np.array([0, 0]), np.array([[0., 0.]])), 25.)
        np.random.seed(3)
        X = np.array([[0., 0.], [0., 1.], [10., 10.], [10., 11.]])
        model.fit(X)
        self.assertTrue(np.all(np.diff(model.errors_) <= 1e-9))
        self.assertTrue(np.isfinite(model.error(X, model.predict(X), model.means)))


if __name__ == '__main__':
    unittest.main()
