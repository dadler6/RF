"""
RandomForest.py

Written by Dan Adler
Email: (daadler0309@gmail.com)
GitHub: https://github.com/dadler6/

Self-implementation of a random forest.

Package requirements:
numpy
pandas
DecisionTree.py (self implementation)
"""

# Imports
import numpy as np
import DecisionTree as DT


class RandomForest(object):
    """
    Abstract decision tree class.  Will use a linked-list-esque implementation where node is a "Node" class,
    and the node holds a cutoff, and then reference to another node.  Nodes can also hold a terminating value.
    Abstraction is made using the private (_) to not be able to implement outside.

    Parameters:


    Methods:
        Public
        ------
        Initialization: Initializes the class


        Private
        -------
    """

    def __init__(self, samp_size=None, num_trees=10, split_type='gini', terminate='leaf', leaf_terminate=1, oob=False):
        """
        Initialize the RandomForest class.

        :param samp_size: The number of samples to put within each decision tree
        :param num_trees: The number of trees to make
        :param split_type: The criteria to split on
        :param terminate: The termination criteria
        :param leaf_terminate: The number of samples to put into a leaf
        :param oob: Whether to cross-validated using an out-of-bag samp.e
        """
        # Set parameters
        self._samp_size = samp_size
        self._num_trees = num_trees
        self._split_type = split_type
        self._terminate = terminate
        self._leaf_terminate = leaf_terminate
        self._oob = oob
        self._trees = []
        self._oob_errors = []

    def fit(self, x_data, y_data):
        """
        Fit (train) a Random Forest model to the data.

        :param x_data: The dataset to train the decision tree with.
        :param y_data: The result vector we are regressing on.
        """
        # Make the number of trees determinate by self._num_trees
        for i in range(self._num_trees):
            # Get tree
            x_in, y_in, x_out, y_out = self.__get_sample(x_data, y_data)
            cdt = self.__get_tree(x_in, y_in)
            # Calculate oob if necessary
            if self._oob:
                self._oob_errors.append(self.__calculate_oob_error(cdt, x_out, y_out))

    def __get_sample(self, x, y):
        """
        Get a sample from two indices.

        :param x: The x data to sample from
        :param y: The y data to sample from
        :return: The sampled x data, y data and the out of sample x data/y data
        """
        # Take the random sample
        idx = np.random.choice(np.arange(len(y)), size=np.floor(self._samp_size * len(y)), replace=True)
        mask = np.ones(len(y), dtype=bool)
        mask[idx] = False
        x_in = x[idx, :]
        y_in = y[idx]
        x_out = x[mask, :]
        y_out = y[mask]

        return x_in, y_in, x_out, y_out

    def __get_tree(self, x, y):
        """
        Create a decision tree based upon self._num_trees.

        :param x: The x data to fit to (input)
        :paray y: The y data to fit to (target)
        :return: A new CDT
        """
        dt = DT.ClassificationDecisionTree(self._split_type, self._terminate,  self._leaf_terminate, prune=False)
        dt.fit(x, y)
        return dt

    @staticmethod
    def __calculate_oob_error(self, cdt, x_out, y_out):
        """
        Calculate the oob error for a tree by predicting on the out of bag sample.

        :param cdt: The fit decision tree
        :param x_out: The out of bag input
        :param y_out: THe out of bad target
        """
        y_pred = cdt.predict(x_out)
        return (np.mean(y_out) - np.mean(y_pred))**2

    def get_oob_error(self):
        pass

    def predict(self, x_data):
        """
        Predict the y (target) for this x_data
        :param x_data: The daata to predict off of
        :return: The predicted target data (y)
        """
        pass
