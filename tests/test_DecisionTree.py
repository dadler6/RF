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
from collections import Counter
import numpy as np
import unittest
import sys

# Import the package
sys.path.insert(0, '../src')
import DecisionTree as DT


class TestRegressionDecisionTreeFit(unittest.TestCase):
    """
    Test the regression decision tree fit class.
    """

    def setUp(self):
        """
        Setup internal parameters used multiple times.
        """
        # Create decision with leaf size as 1
        self.leaf_terminate_1 = 1
        dt_1 = DT.RegressionDecisionTree(
            split_type='rss',
            leaf_terminate=self.leaf_terminate_1
        )

        # Create decision tree with leaf size as 2
        self.leaf_terminate_2 = 2
        dt_2 = DT.RegressionDecisionTree(
            split_type='rss',
            leaf_terminate=self.leaf_terminate_2
        )

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


class TestRegressionDecisionTreePredict(unittest.TestCase):
    """
    Test the regression decision tree fit class.
    """

    def setUp(self):
        """
        Setup internal parameters used multiple times.
        """
        # Create decision with leaf size as 1
        self.leaf_terminate_1 = 1
        self.dt_1 = DT.RegressionDecisionTree(
            split_type='rss',
            leaf_terminate=self.leaf_terminate_1
        )

        # Create decision tree with leaf size as 2
        self.leaf_terminate_2 = 2
        self.dt_2 = DT.RegressionDecisionTree(
            split_type='rss',
            leaf_terminate=self.leaf_terminate_2
        )

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
        true_pred_1 = 0.0
        self.assertEqual(round(result_pred_1, 6), round(true_pred_1, 6))

        # Tree with two leaves
        result_pred_2 = self.dt_2.predict(test_x_data_3)
        true_pred_2 = 0.0
        self.assertEqual(round(result_pred_2, 6), round(true_pred_2, 6))


class TestClassificationDecisionTreeFit(unittest.TestCase):
    """
    Test the classification decision tree fit class.
    """

    def setUp(self):
        """
        Setup internal parameters used multiple times.
        """
        # Create decision with leaf size as 1
        self.leaf_terminate_1 = 1
        dt_1 = DT.ClassificationDecisionTree(
            split_type='gini',
            terminate='leaf',
            leaf_terminate=self.leaf_terminate_1
        )

        # Create decision tree with leaf size as 2
        self.leaf_terminate_2 = 2
        dt_2 = DT.ClassificationDecisionTree(
            split_type='gini',
            terminate='leaf',
            leaf_terminate=self.leaf_terminate_2
        )

        # Create decision tree with leaf pure termination criteria
        dt_3_pure = DT.ClassificationDecisionTree(
            split_type='gini',
            terminate='pure',
        )

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
        self.y_data_1 = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])

        # Train the data
        dt_1.fit(self.x_data_1, self.y_data_1)
        dt_2.fit(self.x_data_1, self.y_data_1)
        dt_3_pure.fit(self.x_data_1, self.y_data_1)

        # Get the result object
        self.result_tree_1 = dt_1.get_tree()
        self.result_tree_2 = dt_2.get_tree()
        self.result_tree_3 = dt_3_pure.get_tree()

    def test_class_values_leaf_terminate(self):
        """
        Test the majority values represent the leaves, and that the termination
        criteria is reached.
        """
        for level in self.result_tree_1:
            for n in level:
                if n.is_leaf():
                    temp_x = n.get_x_data()
                    pred = n.get_prediction()
                    idx = np.unique(np.where((self.x_data_1 == temp_x[0]).all(axis=1))[0])
                    true_class = Counter(self.y_data_1[idx]).most_common(1)[0][0]
                    self.assertEqual(pred, true_class)

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
                    true_class = Counter(self.y_data_1[idx]).most_common(1)[0][0]
                    self.assertEqual(pred, true_class)

    def test_class_values_pure_terminate(self):
        """
        Test that leaves represent one single class, or that the x data size is one.
        """
        for level in self.result_tree_3:
            for n in level:
                if n.is_leaf():
                    temp_x = n.get_x_data()
                    temp_y = n.get_y_data()
                    pred = n.get_prediction()
                    idx = np.array([])
                    # Assert that either y data is shape 1, or that the x data is shape 1
                    self.assertTrue((len(np.unique(temp_y)) == 1) or (temp_x.shape[0] == 1))
                    for i in range(temp_x.shape[0]):
                        r = temp_x[i, :]
                        new = np.unique(np.where((self.x_data_1 == r).all(axis=1))[0])
                        idx = np.concatenate((idx, new))
                    idx = [int(i) for i in idx]
                    true_class = Counter(self.y_data_1[idx]).most_common(1)[0][0]
                    self.assertEqual(pred, true_class)


class TestClassificationDecisionTreeFitGain(unittest.TestCase):
    """
    Test the classification decision tree fit class.
    """

    def setUp(self):
        """
        Setup internal parameters used multiple times.
        """
        # Create decision with tree with a gain ratio

        # Create decision tree with leaf pure termination criteria
        dt_1 = DT.ClassificationDecisionTree(
            split_type='gain_ratio',
            terminate='pure',
        )
        dt_2 = DT.ClassificationDecisionTree(
            split_type='gini',
            terminate='pure',
        )

        # Make simple input data
        self.x_data_1 = np.array([
            [1, 1],
            [2, 1],
            [3, 1],
            [4, 1],
            [5, 1],
            [6, 2],
            [7, 2],
            [8, 2],
            [9, 2],
            [10, 2]
        ])
        self.y_data_1 = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])

        # Train the data
        dt_1.fit(self.x_data_1, self.y_data_1)
        dt_2.fit(self.x_data_1, self.y_data_1)

        # Get the result object
        self.result_tree_1 = dt_1.get_tree()
        self.result_tree_2 = dt_2.get_tree()

    def test_class_values_pure_terminate(self):
        """
        Test that leaves represent one single class, or that the x data size is one.
        """
        for level in self.result_tree_1:
            for n in level:
                if n.is_leaf():
                    temp_x = n.get_x_data()
                    temp_y = n.get_y_data()
                    pred = n.get_prediction()
                    idx = np.array([])
                    # Assert that either y data is shape 1, or that the x data is shape 1
                    self.assertTrue((len(np.unique(temp_y)) == 1) or (temp_x.shape[0] == 1))
                    for i in range(temp_x.shape[0]):
                        r = temp_x[i, :]
                        new = np.unique(np.where((self.x_data_1 == r).all(axis=1))[0])
                        idx = np.concatenate((idx, new))
                    idx = [int(i) for i in idx]
                    true_class = Counter(self.y_data_1[idx]).most_common(1)[0][0]
                    self.assertEqual(pred, true_class)

    def test_gain_ratio_axis(self):
        """
        Test the gain ratio picks the less varied axis more time than the original.
        """
        split_axis_1 = []
        for level in self.result_tree_1:
            for n in level:
                if not n.is_leaf():
                    split_axis_1.append(n.get_col())

        split_axis_2 = []
        for level in self.result_tree_2:
            for n in level:
                if not n.is_leaf():
                    split_axis_2.append(n.get_col())

        bin_count_1 = np.bincount(split_axis_1, minlength=2)
        bin_count_2 = np.bincount(split_axis_2, minlength=2)

        self.assertGreater(bin_count_1[1], bin_count_2[1])


class TestClassificationDecisionTreePredict(unittest.TestCase):
    """
    Test the classification decision tree fit class.
    """

    def setUp(self):
        """
        Setup internal parameters used multiple times.
        """
        # Create decision with leaf size as 1
        self.leaf_terminate_1 = 1
        self.dt_1 = DT.ClassificationDecisionTree(
            split_type='gini',
            terminate='leaf',
            leaf_terminate=self.leaf_terminate_1
        )

        # Create decision tree with leaf size as 2
        self.leaf_terminate_2 = 2
        self.dt_2 = DT.ClassificationDecisionTree(
            split_type='gini',
            terminate='leaf',
            leaf_terminate=self.leaf_terminate_2
        )

        # Create decision tree with leaf pure termination criteria
        self.dt_3_pure = DT.ClassificationDecisionTree(
            split_type='gini',
            terminate='pure',
        )

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
        self.y_data_1 = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])

        # Train the data
        self.dt_1.fit(self.x_data_1, self.y_data_1)
        self.dt_2.fit(self.x_data_1, self.y_data_1)
        self.dt_3_pure.fit(self.x_data_1, self.y_data_1)

    def test_fit_1(self):
        """
        Test a basic prediction.
        """
        # Should work for one leaf
        test_x_data_1 = np.array([1, 4])
        result_pred_1 = self.dt_1.predict(test_x_data_1)
        true_pred_1 = 1
        self.assertEqual(round(result_pred_1, 6), round(true_pred_1, 6))

        # Should also work with two leaves (since [1, 4] appears three times
        result_pred_2 = self.dt_2.predict(test_x_data_1)
        self.assertEqual(round(result_pred_2, 6), round(true_pred_1, 6))

        # [1, 4] should pick majority class which 1
        result_pred_3 = self.dt_3_pure.predict(test_x_data_1)
        self.assertEqual(round(result_pred_3, 6), round(true_pred_1, 6))

    def test_fit_2(self):
        """
        Test a prediction of something that only appears once in the
        tree.
        """
        # Tree with one leaf
        test_x_data_2 = np.array([3, 1])
        result_pred_1 = self.dt_1.predict(test_x_data_2)
        true_pred_1 = 0
        self.assertEqual(round(result_pred_1, 6), round(true_pred_1, 6))

        # Tree with two leaves
        result_pred_2 = self.dt_2.predict(test_x_data_2)
        true_pred_2 = 0
        self.assertEqual(round(result_pred_2, 6), round(true_pred_2, 6))

        # Tree with pure pruning
        result_pred_3 = self.dt_3_pure.predict(test_x_data_2)
        true_pred_3 = 0
        self.assertEqual(round(result_pred_3, 6), round(true_pred_3, 6))

    def test_fit_3(self):
        """
        Test a prediction of something that never appears in the tree.
        """
        # Tree with one leaf
        test_x_data_3 = np.array([7, 4])
        result_pred_1 = self.dt_1.predict(test_x_data_3)
        true_pred_1 = 1.0
        self.assertEqual(round(result_pred_1, 6), round(true_pred_1, 6))

        # Tree with two leaves
        result_pred_2 = self.dt_2.predict(test_x_data_3)
        true_pred_2 = 1.0
        self.assertEqual(round(result_pred_2, 6), round(true_pred_2, 6))

        # Tree with two leaves
        result_pred_3 = self.dt_3_pure.predict(test_x_data_3)
        true_pred_3 = 1.0
        self.assertEqual(round(result_pred_3, 6), round(true_pred_3, 6))


class TestClassificationDecisionTreePruning(unittest.TestCase):
    """
    Test whether a tree is correctly pruned.
    """

    def setUp(self):
        """
        Setup internal parameters used multiple times.
        """
        # Create decision with tree with a gain ratio
        self.dt_w_prune = DT.ClassificationDecisionTree(
            split_type='gain_ratio',
            terminate='pure',
            prune=True
        )
        self.dt_wo_prune = DT.ClassificationDecisionTree(
            split_type='gain_ratio',
            terminate='pure',
            prune=False
        )

        # Make simple input data
        x_data_1 = np.array([[1]] * 2 + [[2]] * 3)
        y_data_1 = np.array([0, 1, 0, 1, 1])

        # Train the data
        self.dt_w_prune.fit(x_data_1, y_data_1)
        self.dt_wo_prune.fit(x_data_1, y_data_1)

    def test_extra_level(self):
        """
        Test that pruning reduces the levels of the trees.
        """
        # Get the result object
        result_tree_w_pruning = self.dt_w_prune.get_tree()
        result_tree_wo_pruning = self.dt_wo_prune.get_tree()

        self.assertEqual(len(result_tree_w_pruning), 1)
        self.assertEqual(len(result_tree_wo_pruning), 2)

    def test_prediction_w_pruning(self):
        """
        Test the prediction with/without pruning.
        """
        test_x_data = np.array([[1]])
        pred_w_pruning = self.dt_w_prune.predict(test_x_data)
        pred_wo_pruning = self.dt_wo_prune.predict(test_x_data)

        self.assertEqual(pred_w_pruning, 1)
        self.assertEqual(pred_wo_pruning, 0)


if __name__ == "__main__":
    unittest.main()
