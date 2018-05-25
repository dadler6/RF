"""
test_RandomForest.py

Written by Dan Adler
Email: (daadler0309@gmail.com)
GitHub: https://github.com/dadler6/

Test cases for RandomForest.py

Package requirements:
numpy
pandas
"""

# Imports
from sklearn import datasets
import numpy as np
import pandas as pd
import unittest
import sys

# Import the package
sys.path.insert(0, '../src')
import RandomForest as RF


class TestRandomForestFit(unittest.TestCase):
    """
    Test that we can correctly fit a random forest classifier.
    """
    def setUp(self):
        """
        Setup necessary test parameters.
        """
        iris_data = datasets.load_iris()
        self.X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
        self.Y = pd.get_dummies(iris_data.target)
        self.Y.columns = iris_data.target_names

    def test_fit_structure(self):
        """
        Test the structure of fitting data using the iris data.
        """
        # Initialize RF with default parameters
        rf_1 = RF.RandomForest()

        # Initialize RF with different parameters
        rf_2 = RF.RandomForest(samp_size=0.63, num_trees=20, num_features=2)

        # Initialize RF with oob
        rf_3 = RF.RandomForest(terminate='pure', oob=True)

        # Fit the iris data
        rf_1.fit(self.X, self.Y['setosa'])
        rf_2.fit(self.X, self.Y['setosa'])
        rf_3.fit(self.X, self.Y['setosa'])

        # Get trees
        res_1 = rf_1.get_trees()
        res_2 = rf_2.get_trees()
        res_3 = rf_3.get_trees()

        # Check number of trees
        self.assertEquals(len(res_1), 10)
        self.assertEquals(len(res_2), 20)
        self.assertEquals(len(res_3), 10)


if __name__ == "__main__":
    unittest.main()