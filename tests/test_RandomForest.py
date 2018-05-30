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
        self.assertEqual(len(res_1), 10)
        self.assertEqual(len(res_2), 20)
        self.assertEqual(len(res_3), 10)

        # Get cols used
        cols_used_1 = rf_1.get_cols_used()
        cols_used_2 = rf_2.get_cols_used()
        cols_used_3 = rf_3.get_cols_used()

        # Check lengths are correct
        for c in cols_used_1:
            c.sort()
            self.assertEqual(list(c), [0, 1, 2, 3])

        for c in cols_used_2:
            c.sort()
            self.assertEqual(len(c), 2)
            self.assertEqual(len(set(c) & {0, 1, 2, 3}), 2)

        for c in cols_used_3:
            c.sort()
            self.assertEqual(list(c), [0, 1, 2, 3])

        # Assert that tree 3 has an oob error
        oob_3 = rf_3.get_oob_error()
        self.assertGreaterEqual(oob_3, 0.0)
        self.assertLessEqual(oob_3, 1.0)

    def test_predict(self):
        """
        Test that one can correctly predict obvious answers.
        """
        # Initialize three RFs with oob and pure terminate
        rf_1 = RF.RandomForest(terminate='pure', oob=True)
        rf_2 = RF.RandomForest(terminate='pure', oob=True)
        rf_3 = RF.RandomForest(terminate='pure', oob=True)

        # Fit the rfs
        rf_1.fit(self.X, self.Y['setosa'])
        rf_2.fit(self.X, self.Y['versicolor'])
        rf_3.fit(self.X, self.Y['virginica'])

        # Test with basics
        x_test = pd.DataFrame({
            'sepal length (cm)': [4.5, 5.5, 7.5],
            'sepal width (cm)': [3.5, 2.5, 3.5],
            'petal length (cm)': [1.5, 4.0, 6.0],
            'petal width (cm)': [0.5, 1.25, 2.0]
        })
        x_test = x_test[self.X.columns.values]

        # Predict
        y_pred_1 = rf_1.predict(x_test)
        y_pred_2 = rf_2.predict(x_test)
        y_pred_3 = rf_3.predict(x_test)

        # Assertions
        self.assertEqual(list(y_pred_1), [1, 0, 0])
        self.assertEqual(list(y_pred_2), [0, 1, 0])
        self.assertEqual(list(y_pred_3), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
