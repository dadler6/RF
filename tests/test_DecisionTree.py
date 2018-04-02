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
import pandas as pd
import unittest
import sys

# Import the package
sys.path.insert(0, '../src')
import DecisionTree as DT


class TestDecisionTreeFit(unittest.TestCase):
    """
    Test the decision tree fit class.
    """

    def test_fit_regressor_1(self):
        """
        Test the regressor class with a simple dataset.
        """
        # Create decision with leaf size as 1
        dt_1 = DT.DecisionTree(leaf_terminate=1)

        # Make simple input data
        x_data_1 = np.array([
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
        y_data_1 = np.array([5, 6, 5, 1, 6, 7, 8, 6, 4, 0])

        # Train the data
        dt_1.fit(x_data_1, y_data_1)

        # Get the result object
        result_tree_1 = dt_1.get_tree()

        self.assertEquals(len(result_tree_1), 4)


if __name__ == "__main__":
    unittest.main()




