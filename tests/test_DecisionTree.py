"""
test_DecisionTree.py

Written by Dan Adler
Email: (daadler0309@gmail.com)
GitHub: https://github.com/dadler6/

Test cases for DecisionTree.py

Package requirements:
numpy
"""

# Imports
import numpy as np
import unittest
import sys

# Import the package
sys.path.insert(0, '../src')
import DecisionTree as DT


class TestDecisionTreeFit(unittest.TestCase):
    """
    Test the decision tree fit class.
    """

    def setUp(self):
        """
        Setup internal parameters used multiple times.
        """
        # Create decision with leaf size as 1
        self.leaf_terminate_1 = 1
        dt_1 = DT.DecisionTree(leaf_terminate=self.leaf_terminate_1)

        # Create decision tree with leaf size as 2
        self.leaf_terminate_2 = 2
        dt_2 = DT.DecisionTree(leaf_terminate=self.leaf_terminate_2)

        # Make simple input data
        self.x_data_1 = np.array([
            [1, 4],
            [6, 7],
            [1, 4],
            [2, 3],
            [4, 5],
            [1, 5],
            [3, 6],
            [1, 4],
            [3, 1],
            [8, 9]
        ])
        self.y_data_1 = np.array([5, 6, 5, 1, 6, 7, 8, 6, 4, 0])

        # Train the data
        dt_1.fit(self.x_data_1, self.y_data_1)
        dt_2.fit(self.x_data_1, self.y_data_1)

        # Get the result object
        self.result_tree_1 = dt_1.get_tree()
        self.result_tree_2 = dt_2.get_tree()

    def test_leaf_size(self):
        """
        Test the leaf size is correct.
        """
        for level in self.result_tree_1:
            for n in level:
                if n.is_leaf():
                    temp_x = n.get_x_data()
                    self.assertEqual(temp_x.shape[0], self.leaf_terminate_1)

        for level in self.result_tree_2:
            for n in level:
                if n.is_leaf():
                    temp_x = n.get_x_data()
                    self.assertLessEqual(temp_x.shape[0], self.leaf_terminate_2)

    def test_mean_values(self):
        """
        Test the mean values represent the leaves.
        """
        for level in self.result_tree_1:
            for n in level:
                if n.is_leaf():
                    temp_x = n.get_x_data()
                    pred = n.get_prediction()
                    idx = np.unique(np.where((self.x_data_1 == temp_x[0]).all(axis=1))[0])
                    true_mean = np.mean(self.y_data_1[idx])
                    self.assertEqual(pred, true_mean)

        for level in self.result_tree_2:
            for n in level:
                if n.is_leaf():
                    temp_x = n.get_x_data()
                    pred = n.get_prediction()
                    idx = np.array([])
                    for i in range(temp_x.shape[0]):
                        r = temp_x[i, :]
                        new = np.unique(np.where((self.x_data_1 == r).all(axis=1))[0])
                        idx = np.concatenate((idx, new))
                    idx = [int(i) for i in idx]
                    true_mean = np.mean(self.y_data_1[idx])
                    self.assertEqual(pred, true_mean)


class TestDecisionTreePredict(unittest.TestCase):
    """
    Test the decision tree fit class.
    """

    def setUp(self):
        """
        Setup internal parameters used multiple times.
        """
        # Create decision with leaf size as 1
        self.leaf_terminate_1 = 1
        self.dt_1 = DT.DecisionTree(leaf_terminate=self.leaf_terminate_1)

        # Create decision tree with leaf size as 2
        self.leaf_terminate_2 = 2
        self.dt_2 = DT.DecisionTree(leaf_terminate=self.leaf_terminate_2)

        # Make simple input data
        self.x_data_1 = np.array([
            [1, 4],
            [6, 7],
            [1, 4],
            [2, 3],
            [4, 5],
            [1, 5],
            [3, 6],
            [1, 4],
            [3, 1],
            [8, 9]
        ])
        self.y_data_1 = np.array([5, 6, 5, 1, 6, 7, 8, 6, 4, 0])

        # Train the data
        self.dt_1.fit(self.x_data_1, self.y_data_1)
        self.dt_2.fit(self.x_data_1, self.y_data_1)

    def test_fit_1(self):
        """
        Test a basic prediction.
        """
        # Should work for one leaf
        test_x_data_1 = np.array([1, 4])
        result_pred_1 = self.dt_1.predict(test_x_data_1)
        true_pred_1 = 5.33333333333
        self.assertEqual(round(result_pred_1, 6), round(true_pred_1, 6))

        # Should also work with two leaves (since [1, 4] appears three times
        result_pred_2 = self.dt_2.predict(test_x_data_1)
        self.assertEqual(round(result_pred_2, 6), round(true_pred_1, 6))

    def test_fit_2(self):
        """
        Test a prediction of something that only appears once in the
        tree.
        """
        # Tree with one leaf
        test_x_data_2 = np.array([3, 1])
        result_pred_1 = self.dt_1.predict(test_x_data_2)
        true_pred_1 = 4
        self.assertEqual(round(result_pred_1, 6), round(true_pred_1, 6))

        # Tree with two leaves
        result_pred_2 = self.dt_2.predict(test_x_data_2)
        true_pred_2 = 2.5
        self.assertEqual(round(result_pred_2, 6), round(true_pred_2, 6))

    def test_fit_3(self):
        """
        Test a prediction of something that never appears in the tree.
        """
        # Tree with one leaf
        test_x_data_3 = np.array([7, 4])
        result_pred_1 = self.dt_1.predict(test_x_data_3)
        true_pred_1 = 4
        self.assertEqual(round(result_pred_1, 6), round(true_pred_1, 6))

        # Tree with two leaves
        result_pred_2 = self.dt_2.predict(test_x_data_3)
        true_pred_2 = 2.5
        self.assertEqual(round(result_pred_2, 6), round(true_pred_2, 6))





if __name__ == "__main__":
    unittest.main()
