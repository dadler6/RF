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
        self.__node_list


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
        self.__node_list = []
        self.__ncols = 0

    def fit(self, x_data, y_data):
        """
        Fit (train) the decision tree using an inputted dataset.

        :param x_data: The dataset to train the decision tree with.
        :param y_data: The result vector we are regressing on.
        """
        # Get number of columns
        self.__ncols = x_data.shape[1]

        # Make initial node with all data
        self.__node_list.append([self.__create_node__(x_data, y_data)])
        
        # Recursive fit
        self.__recursive_fit([0, 0])
        
    def __recursive_fit(self, curr_idx):
        """
        Recursively fit nodes while not satisfying the terminating criteria.

        :param curr_idx: The current 2-d index the function is calling to
        """
        if not self.__terminate_fit__(curr_idx):
            # Go through each column
            col_rss = []
            for i in range(self.__ncols):
                col_rss.append(self.__rss__(curr_idx[0], curr_idx[1], i))
            split_col = np.argmin(col_rss)
            # Set the split
            self.__node_list[curr_idx[0]][curr_idx[1]].set_split(split_col)
            # Create new nodes
            lower_idx, upper_idx = self.__create_new_nodes__(curr_idx[0], curr_idx[1])
            # Call the function if necessary
            if lower_idx[1] is not None:
                self.__recursive_fit(lower_idx)
            if upper_idx[1] is not None:
                self.__recursive_fit(upper_idx)

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

    def __create_new_nodes__(self, level, n):
        """
        Create the next level of nodes. Splits the data based upon the specified axis, and
        creates the new level of nodes by splitting the data.

        :param level: The level value to create the new nodes on
        :param n: The index in the level we are on

        :return: the upper and lower tuples for the new nodes created
        """
        if (level + 1) == len(self.__node_list):
            self.__node_list.append([])

        split_val = self.__node_list[level][n].get_split()
        x_data = self.__node_list[level][n].get_x_data()
        y_data = self.__node_list[level][n].get_y_data()
        idx = self.__node_list[level][n].get_col()

        # Split data
        lower_x_data = x_data[x_data[:, idx] < split_val]
        lower_y_data = y_data[x_data[:, idx] < split_val]
        upper_x_data = x_data[x_data[:, idx] >= split_val]
        upper_y_data = y_data[x_data[:, idx] >= split_val]

        # Now check if all the same in lower/upper
        # Do not change y_data to average over all values
        if (lower_x_data.shape[0] > 1) and ((lower_x_data - lower_x_data[0, :]) == 0).all():
            lower_x_data = lower_x_data[[0], :]
        if (upper_x_data.shape[0] > 1) and ((upper_x_data - upper_x_data[0, :]) == 0).all():
            upper_x_data = upper_x_data[[0], :]

        # Make lower node if one can
        if lower_x_data.shape[0] > 0:
            lower_curr_index = len(self.__node_list[level + 1])
            self.__node_list[level + 1].append(self.__create_node__(lower_x_data, lower_y_data))
            self.__node_list[level][n].set_lower_split_index(lower_curr_index)
        else:
            lower_curr_index = None
        # Make upper node
        if upper_x_data.shape[0] > 0:
            upper_curr_index = len(self.__node_list[level + 1])
            self.__node_list[level + 1].append(self.__create_node__(upper_x_data, upper_y_data))
            self.__node_list[level][n].set_lower_split_index(upper_curr_index)
        else:
            upper_curr_index = None

        return [level + 1, lower_curr_index], [level + 1, upper_curr_index]

    def __rss__(self, level, n, idx):
        """
        Calculates the residual sum of square errors for a specific region.

        :param level: The level in the dictionary to look at
        :param n: The node index to calculate the rss for
        :param idx: The column index in the matrix to calculate values for

        :return: The RSS
        """
        curr = self.__node_list[level][n].get_x_data()[:, idx]
        # Make sure not to split on axis with the same data
        if len(np.unique(curr)) == 1:
            return 1e10
        else:
            return np.sum((curr - np.mean(curr))**2)

    def __terminate_fit__(self, curr_idx):
        """
        Decide if fit is terminated.

        :param: The current 2D idx
        :return: True if terminated, False if not
        """
        if self.__node_list[curr_idx[0]][curr_idx[1]].is_leaf():
            return True
        return False

    def predict(self, x_data):
        """
        Predict a class using the dataset given.

        :param x_data: The dataset to predict
        :return: A vector of predictions for each row in X.
        """
        pass

    def get_tree(self):
        """
        Get the underlying tree object.

        :return: The tree (self.__node_list())
        """
        return self.__node_list

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
            self.__split = np.mean(self.__x_data[:, idx])

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
