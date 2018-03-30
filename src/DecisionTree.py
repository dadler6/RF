"""
DecisionTree.py

Written by Dan Adler
Email: (daadler0309@gmail.com)
GitHub: https://github.com/dadler6/

Self-implementation of a decision tree.

Package requirements:
numpy
"""

# Imports
import numpy as np


class DecisionTree(object):
    """
    Decision tree class.  Will use a linked-list-esque implementation where node is a "Node" class, and the node holds
    a cutoff, and then reference to another node.  Nodes can also hold a terminating value.

    Parameters:
        self.leaf_terminate: The leaf terminating criteria (how many points to left in the data to create
                             a leaf node). Defaults to 1.


    Methods:
        Public
        ------
        Initialization: initialize the decision tree. Choose the number of terminating parameters for the tree.
        fit: Takes an inputted dataset and creates the decision tree.
        predict: Takes a new dataset, and runs the algorithm to perform a prediction.

        Private
        -------
        Node Class: A node class (see node class for explanation)
        __rss__: Calculates the residual sum of squares to split regions
    """

    def __init__(self, leaf_terminate=1):
        """
        Initialize the decision tree.

        :param leaf_terminate: the amount of collections needed to terminate the tree with a leaf (defaults to 1)
        """
        self.leaf_terminate = leaf_terminate
        self.node_dict = {}

    def fit(self, x_data):
        """
        Fit (train) the decision tree using an inputted dataset.

        :param x_data: The dataset to train the decision tree with.
        """
        # Make initial node with all data
        # Determine RSS
        # Split (create node)

        # While not terminate fit
        # Go to next level
        # Determine RSS
        # Split
        pass

    def __rss__(self, level, n, idx):
        """
        Calculates the residual sum of square errors for a specific region.

        :param level: The level in the dictionary to look at
        :param n: The node index to calculate the rss for
        :param idx: The column index in the matrix to calculate values for

        :return: The RSS
        """
        curr = self.node_dict[level][n].get_data()[idx]
        return np.sum((curr - np.mean(curr))**2)

    def __sum_rss__(self, level, idx):
        """
        Calculate the average of each node list value for the given param index idx

        :param level: The level to calculate the summed rss
        :param idx: The index of the column the data frame the summed rss is for

        :return: The summed RSS
        """
        return np.sum([self.__rss__(level, n, idx) for n in range(len(self.node_dict))])

    def __terminate_fit__(self, level):
        """
        Decide if fit is terminated.

        :param: The current level
        :return: True if terminated, False if not
        """
        for i in self.node_dict[level]:
            if not i.get_leaf():
                return False
        return True

    def predict(self, x_data):
        """
        Predict a class using the dataset given.

        :param x_data: The dataset to predict
        :return: A vector of predictions for each row in X.
        """
        pass

    class _Node(object):
        """
        Internal node class.  Used to hold splitting values, or termination criteria.
        All parameters are private since we do not any editing to take place after we setup the node.

        Parameters:
            self.__leaf__: True if leaf node/False if not
            self.__data__: The array of data points for that node
            self.__split__: The splitting criteria, or prediction value (if this is a leaf)
            self.__lower_split__: If a non-leaf node, the reference to the place in the next level list for
                                  the coming node
            self.__upper_split__: If a non-leaf node, the reference to the place in the next level list
                                  for the coming node
        """

        def __init__(self, leaf, data):
            """
            Initialize the node.

            :param leaf: The true/false value (defaults to false) to say if a node is a leaf
            :param data: The data to be placed into the node
            """
            # Set leaf value
            self.__leaf__ = leaf
            self.__data__ = data

            # Set other values to one
            self.__split__ = np.mean(data)

            # Set other values to None
            self.__lower_split__ = None
            self.__upper_split__ = None

        def is_leaf(self):
            """
            Return self.__leaf__

            :return: self.__leaf__ value
            """
            return self.__leaf__

        def get_data(self):
            """
            Return the data for this node (self.__data__)

            :return: self.__data__
            """
            return self.__data__

        def get_prediction(self):
            """
            Return the prediction (if it is a leaf)

            :return: self.__prediction__
            """
            return self.__split__

        def set_lower_split_index(self, idx):
            """
            Set the lower split value.

            :param idx: the index of the lower split
            """
            self.__lower_split__ = idx

        def set_upper_split_index(self, idx):
            """
            Set the lower split value.

            :param idx: the index of the upper split
            """
            self.__upper_split__ = idx

        def get_lower_split(self):
            """
            Get the value for index to the lower split data (if non-leaf node)

            :return: self.__lower_split__
            """
            return self.__lower_split__

        def get_upper_split(self):
            """
            Get the value for the index to the upper split data (if non-leaf node)

            :return: self.__upper_split__
            """
            return self.__upper_split__
