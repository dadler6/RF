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
        self.__leaf_terminate: The leaf terminating criteria (how many points to left in the data to create
                             a leaf node). Defaults to 1.
        self.__node_dict


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
        self.__leaf_terminate = leaf_terminate
        self.__node_dict = {}

    def fit(self, x_data, y_data):
        """
        Fit (train) the decision tree using an inputted dataset.

        :param x_data: The dataset to train the decision tree with.
        :param y_data: The result vector we are regressing on.
        """
        # Level parameter
        level = 0
        # Get number of columns
        ncols = x_data.shape[1]

        # Make initial node with all data
        self.__node_dict[0] = [self.__create_node__(x_data, y_data)]

        # While not terminate fit
        while not self.__terminate_fit__(level):
            # Determine RSS for each index
            col_rss = []
            for idx in range(ncols):
                col_rss.append(self.__sum_rss__(level, idx))
            # Get the minimum RSS value
            min_idx = np.argmin(col_rss)
            # Split and create new nodes
            self.__add_split_value__(min_idx, level)
            # Create new node levels
            self.__create_new_nodes__(level)
            level += 1

    def __create_node__(self, x_data, y_data):
        """
        Creates new node and determines if it is a leaf node.

        :param x_data: The x data to create the node
        :param y_data: The prediction data to create the node
        :return: The new node object
        """
        if x_data.shape[0] > self.__leaf_terminate:
            return self._Node(False, x_data, y_data)
        else:
            return self._Node(True, x_data, y_data)

    def __add_split_value__(self, idx, level):
        """
        Split nodes along an index value.

        :param idx: The index to split on
        :param level: The level in the dictionary currently being set
        """
        self.__node_dict[level] = [n.set_split(idx) for n in self.__node_dict[level] if not n.is_leaf()]

    def __create_new_nodes__(self, level):
        """
        Create the next level of nodes. Splits the data based upon the specified axis, and
        creates the new level of nodes by splitting the data.

        :param level: The level value to create the new nodes on
        """
        self.__node_dict[level + 1] = []
        curr_index = 0
        for i in range(len(self.__node_dict[level])):
            split_val = self.__node_dict[level][i].get_split()
            data = self.__node_dict[level][i].get_x_data()
            y_data = self.__node_dict[level][i].get_y_data()
            idx = self.__node_dict[level][i].get_col()
            # Split data
            lower_x_data = data[data[idx] < split_val]
            lower_y_data = y_data[data[idx] < split_val]
            upper_x_data = data[data[idx] >= split_val]
            upper_y_data = y_data[data[idx] >= split_val]
            # Make lower node
            self.__node_dict[level + 1].append(self.__create_node__(lower_x_data, lower_y_data))
            self.__node_dict[level][i].set_lower_split_index(curr_index)
            curr_index += 1
            # Make upper node
            self.__node_dict[level + 1].append(self.__create_node__(upper_x_data, upper_y_data))
            self.__node_dict[level][i].set_lower_split_index(curr_index)
            curr_index += 1

    def __rss__(self, level, n, idx):
        """
        Calculates the residual sum of square errors for a specific region.

        :param level: The level in the dictionary to look at
        :param n: The node index to calculate the rss for
        :param idx: The column index in the matrix to calculate values for

        :return: The RSS
        """
        curr = self.__node_dict[level][n].get_x_data()[idx]
        return np.sum((curr - np.mean(curr))**2)

    def __sum_rss__(self, level, idx):
        """
        Calculate the average of each node list value for the given param index idx

        :param level: The level to calculate the summed rss
        :param idx: The index of the column the data frame the summed rss is for

        :return: The summed RSS
        """
        return np.sum([self.__rss__(level, n, idx) for n in range(len(self.__node_dict[level]))])

    def __terminate_fit__(self, level):
        """
        Decide if fit is terminated.

        :param: The current level
        :return: True if terminated, False if not
        """
        for i in self.__node_dict[level]:
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
            self.__leaf: True if leaf node/False if not
            self.__data: The array of data points for that node
            self.__split: The splitting criteria. if it is not a leaf
            self.__prediction: The predictionv laue, if it is a leaf
            self.__col: The column one is splitting on
            self.__lower_split: If a non-leaf node, the reference to the place in the next level list for
                                  the coming node
            self.__upper_split: If a non-leaf node, the reference to the place in the next level list
                                  for the coming node
        """

        def __init__(self, leaf, x_data, y_data):
            """
            Initialize the node.

            :param leaf: The true/false value (defaults to false) to say if a node is a leaf
            :param x_data: The data to be placed into the node
            :param y_data: The y_data to be averaged over if a leaf node
            """
            # Set leaf value
            self.__leaf = leaf
            self.__x_data = x_data
            self.__y_data = y_data

            # If a leaf, take average of y_data
            if self.__leaf:
                self.__prediction = np.mean(y_data)
            else:
                self.__prediction = None

            # Set other values to None
            self.__lower_split = None
            self.__upper_split = None
            self.__col = None
            self.__split = None

        def is_leaf(self):
            """
            Return self.__leaf__

            :return: self.__leaf__ value
            """
            return self.__leaf

        def set_split(self, idx):
            """
            Set the column/split index this node splits on.  Also
            sets the split value for a non-leaf node.

            :param idx: The index
            """
            self.__col = idx
            self.__split = np.mean(self.__x_data[idx])

        def set_lower_split_index(self, idx):
            """
            Set the lower split value.

            :param idx: the index of the lower split
            """
            self.__lower_split = idx

        def set_upper_split_index(self, idx):
            """
            Set the lower split value.

            :param idx: the index of the upper split
            """
            self.__upper_split = idx

        def get_x_data(self):
            """
            Return the x_data for this node (self.__data__)

            :return: self.__x_data
            """
            return self.__x_data

        def get_y_data(self):
            """
            Return the y_data for this node (self.__data__)

            :return: self.__y_data
            """
            return self.__y_data

        def get_prediction(self):
            """
            Return the prediction (if it is a leaf)

            :return: self.__split
            """
            return self.__prediction

        def get_col(self):
            """
            Get the column index the node splits on.

            :return: The column index
            """
            return self.__col

        def get_split(self):
            """
            Get the split value.

            :return: The split value
            """
            return self.__split

        def get_lower_split(self):
            """
            Get the value for index to the lower split data (if non-leaf node)

            :return: self.__lower_split
            """
            return self.__lower_split

        def get_upper_split(self):
            """
            Get the value for the index to the upper split data (if non-leaf node)

            :return: self.__upper_split
            """
            return self.__upper_split
