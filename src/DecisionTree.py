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
from collections import Counter


class DecisionTree(object):
    """
    Abstract decision tree class.  Will use a linked-list-esque implementation where node is a "Node" class,
    and the node holds a cutoff, and then reference to another node.  Nodes can also hold a terminating value.

    Parameters:
        self.__leaf_terminate: The leaf terminating criteria (how many points to left in the data to create
                             a leaf node). Defaults to 1 or None (if termination criteria = 'leaf'.
        self.__pure_terminate: True/False pending if the termination criteria is pure for classification trees
        self.__node_list: A list of 2D nodes, where each section of the outer list is a level of the
                          tree, and then the lists within a level are the inidividual nodes at that level
        self.__ncols: The number of features within the given dataset.
        self.__type: classification or regression
        self.__split_type: type of splitting


    Methods:
        Public
        ------
        Initialization: Initializes the class
        fit: Takes an inputted dataset and creates the decision tree.
        predict: Takes a new dataset, and runs the algorithm to perform a prediction.
        get_tree: Returns the tree with leaf nodes (self.__node_list)

        Private
        -------
        Node Class: A node class (see node class for explanation)
        __recursive_fit__: Fits a tree recursively by calculating a column to split on
        __create_node__: Creates a new node, and designs if it's a leaf
        __create_new_nodes__: Create a set of new nodes by splitting
        __rss__: Calculates the residual sum of squares to split regions
        __terminate_fit__: Checks at a stage whether each leaf satisfies the terminating criteria
        __recursive_predict__: Does the recurisive predictions at each point
    """

    def __init__(self, tree_type, split_type, terminate='leaf', leaf_terminate=None):
        """
        Initialize the decision tree.
        :param tree_type: either classification or regression
        :param split_type: the criterion to split a node (either rss, gini, gain_ratio)
        :param terminate: the termination criteria (defaults to leaf, can be 'pure' if classification)
        :param tree_type: the type of decision tree (classification or regression)
        """
        # Check if this is a base class
        if self.__class__.__name__ == 'DecisionTree':
            raise TypeError('Cannot instantiate base class DecisionTree')
        # Check to make sure that split type is not 'gini' for regression
        elif self.__class__.__name__ == 'RegressionDecisionTree':
            if split_type == 'gini':
                raise ValueError('Cannot have split_type=gini for class RegressionDecisionTree')
            elif split_type == 'gain_ratio':
                raise ValueError('Cannot have split_type=gain_ratio for class RegressionDecisionTree')
            if terminate == 'pure':
                raise ValueError('Cannot have a pure termination for class RegressionDecisionTree')
        else:  # Will default to this is a ClassificationDecisionTree
            if split_type == 'rss':
                raise ValueError('Cannot have split_type=rss for class ClassificationDecisionTree')
        # Check termination criteria
        if terminate == 'leaf':
            if (leaf_terminate is None) or leaf_terminate < 1:
                raise ValueError('Cannot have non-positive termination criteria for terminate == "leaf"')
            self.__leaf_terminate = leaf_terminate
            self.__pure_terminate = False
        else:
            self.__leaf_terminate = None
            self.__pure_terminate = True
        self.__node_list = []
        self.__ncols = 0
        self.__type = tree_type
        self.__split_type = split_type

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
        self.__recursive_fit__([0, 0])
        
    def __recursive_fit__(self, curr_idx):
        """
        Recursively fit nodes while not satisfying the terminating criteria.

        :param curr_idx: The current 2-d index the function is calling to
        """
        level, n = curr_idx[0], curr_idx[1]
        if not self.__terminate_fit__(curr_idx):
            # Go through each column
            if self.__split_type == 'rss':
                split_col, val = self.__find_split__(curr_idx, np.min, np.argmin, self.__rss__)
                # Set the split
            elif self.__split_type == 'gini':
                split_col, val = self.__find_split__(curr_idx, np.max, np.argmax, self.__gini_impurity_gain__)
            elif self.__split_type == 'gain_ratio':
                split_col, val = self.__find_split__(curr_idx, np.max, np.argmax, self.__gain_ratio__)
            else:
                raise ValueError('Unknown split type defined')
            self.__node_list[level][n].set_split(split_col, val)
            # Create new nodes
            lower_idx, upper_idx = self.__create_new_nodes__(level, n)
            # Call the function if necessary
            if lower_idx[1] is not None:
                self.__recursive_fit__(lower_idx)
            if upper_idx[1] is not None:
                self.__recursive_fit__(upper_idx)

    def __create_node__(self, x_data, y_data):
        """
        Creates new node and determines if it is a leaf node.

        :param x_data: The x data to create the node
        :param y_data: The prediction data to create the node
        :return: The new node object
        """
        # Return if leaf
        if self.__pure_terminate:
            # Check y_data holds one unique value
            if len(np.unique(y_data)) > 1 and x_data.shape[0] > 1:
                return self._Node(False, x_data, y_data)
        else:
            # Check leaf size
            if x_data.shape[0] > self.__leaf_terminate:
                return self._Node(False, x_data, y_data)

        # Return if branching node
        if self.__type == 'classification':
            return self._Node(True, x_data, y_data, 'classification')
        else:
            return self._Node(True, x_data, y_data, 'regression')

    def __split_data__(self, level, n, idx, split_val):
        """
        Split the data based upon a value.

        :param level: the level
        :param n: the node index in the level
        :param idx: the index to split on
        :param split_val: the split value

        :return: the split
        """
        x_data = self.__node_list[level][n].get_x_data()
        y_data = self.__node_list[level][n].get_y_data()

        lower_x_data = x_data[x_data[:, idx] < split_val]
        lower_y_data = y_data[x_data[:, idx] < split_val]
        upper_x_data = x_data[x_data[:, idx] >= split_val]
        upper_y_data = y_data[x_data[:, idx] >= split_val]

        return lower_x_data, lower_y_data, upper_x_data, upper_y_data

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
        idx = self.__node_list[level][n].get_col()

        # Split data
        lower_x_data, lower_y_data, upper_x_data, upper_y_data = self.__split_data__(level, n, idx, split_val)

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
            self.__node_list[level][n].set_upper_split_index(upper_curr_index)
        else:
            upper_curr_index = None

        return [level + 1, lower_curr_index], [level + 1, upper_curr_index]

    def __find_split__(self, curr_idx, decision_func, arg_func, criteria_func):
        """
        Find split using the given criteria

        :param curr_idx: The current 2-d index the function is calling to
        :param decision_func: np.min/np.max depending if rss vs. gini
        :param arg_func: np.argmin/np.argmax depending if rss vs. gini
        :param criteria_func: either self.__rss__, self.__gini_impurity_gain__, or self.__gain_ratio__

        :return: the split column/value
        """
        level, n = curr_idx[0], curr_idx[1]
        x_data = self.__node_list[level][n].get_x_data()
        col_min = []
        col_val = []
        for i in range(self.__ncols):
            temp_desc = []
            temp_val = []
            temp_list = list(np.unique(x_data[:, i]))
            temp_list.sort()
            for j in range(len(temp_list) - 1):
                m = np.mean([temp_list[j], temp_list[j + 1]])
                temp_val.append(m)
                temp_desc.append(criteria_func(curr_idx[0], curr_idx[1], i, m))
            # Checks
            if len(temp_desc) == 0:
                if decision_func == np.min:
                    temp_desc.append(1e10)
                else:
                    temp_desc.append(-1e10)
                temp_val.append(0)
            col_min.append(decision_func(temp_desc))
            col_val.append(temp_val[arg_func(temp_desc)])

        return arg_func(col_min), col_val[arg_func(col_min)]

    def __rss__(self, level, n, idx, split_val):
        """
        Calculates the residual sum of square errors for a specific region.

        :param level: The level in the dictionary to look at
        :param n: The node index to calculate the rss for
        :param idx: The column index in the matrix to calculate values for
        :param split_val: The value to split on

        :return: The RSS
        """
        _, lower_y_data, _, upper_y_data = self.__split_data__(level, n, idx, split_val)
        return np.sum((lower_y_data - np.mean(lower_y_data))**2) + np.sum((upper_y_data - np.mean(upper_y_data))**2)

    @staticmethod
    def __gini_impurity__(y_data):
        """
        Calculate the gini impurity (1 - sum(p(i)^2)

        :param y_data: the y data

        :return: the impurity
        """
        _, counts = np.unique(y_data, return_counts=True)
        return 1 - np.sum((counts / np.sum(counts))**2)

    @staticmethod
    def __split_information__(x):
        """
        Calculate the gain ratio (-sum(|S_i|/|S| * log_2(|S_i|/|S|))

        :param x: the specific x vector

        :return: the split information for that variable
        """
        freq = np.bincount(x) / len(x)
        return np.sum([-1 * i * np.log2(i) for i in freq if i > 0.0])

    def __gini_impurity_gain__(self, level, n, idx, split_val):
        """
        Calculates the gain in gini impurity for a specific region.
        Should ONLY be used in classification problems.
        Gain = Curr Gini * Size - sum_{new nodes}(new_gini * size)

        :param level: The level in the dictionary to look at
        :param n: The node index to calculate the rss for
        :param idx: The column index in the matrix to calculate values for
        :param split_val: The value to split on

        :return: The gini impurity for this split
        """
        y_data = self.__node_list[level][n].get_y_data()
        _, lower_y_data, _, upper_y_data = self.__split_data__(level, n, idx, split_val)
        curr = self.__gini_impurity__(y_data)*len(y_data)
        lower = self.__gini_impurity__(lower_y_data)*len(lower_y_data)
        upper = self.__gini_impurity__(upper_y_data)*len(upper_y_data)
        return curr - (lower + upper)

    def __gain_ratio__(self, level, n, idx, split_val):
        """
        Calculates the gain ratio, which is equal to the (impurity gain)/(split information)
        Should ONLY be used in classification problems.

        :param level: The level in the dictionary to look at
        :param n: The node index to calculate the rss for
        :param idx: The column index in the matrix to calculate values for
        :param split_val: The value to split on

        :return: The gain ratio
        """
        x_data = self.__node_list[level][n].get_x_data()
        return self.__gini_impurity_gain__(level, n, idx, split_val) / self.__split_information__(x_data[:, idx])

    def __terminate_fit__(self, curr_idx):
        """
        Decide if fit is terminated.

        :param: The current 2D idx
        :return: True if terminated, False if not
        """
        if self.__node_list[curr_idx[0]][curr_idx[1]].is_leaf():
            return True
        return False

    def __recursive_predict__(self, x_data, curr_idx):
        """
        Follow the tree to get the correct prediction.

        :param x_data: The data we are predicting on.
        :param curr_idx: The current node we are looking at
        :return: The prediction
        """
        # Check if leaf
        if self.__node_list[curr_idx[0]][curr_idx[1]].is_leaf():
            return self.__node_list[curr_idx[0]][curr_idx[1]].get_prediction()
        else:
            # Figure out next leaf to look at
            idx = self.__node_list[curr_idx[0]][curr_idx[1]].get_col()
            split = self.__node_list[curr_idx[0]][curr_idx[1]].get_split()
            if x_data[idx] < split:
                new_idx = [curr_idx[0] + 1, self.__node_list[curr_idx[0]][curr_idx[1]].get_lower_split()]
            else:
                new_idx = [curr_idx[0] + 1, self.__node_list[curr_idx[0]][curr_idx[1]].get_upper_split()]
            return self.__recursive_predict__(x_data, new_idx)

    def predict(self, x_data):
        """
        Predict a class using the dataset given.

        :param x_data: The dataset to predict
        :return: A vector of predictions for each row in X.
        """
        return self.__recursive_predict__(x_data, [0, 0])

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

        Methods:
        Public
        ------
        Initialization: Initialize a new node, and determine if it is a leaf
        is_leaf: True/false statement returning if this is a leaf
        predict: Takes a new dataset, and runs the algorithm to perform a prediction.
        set_split: Set the split amount to branch the tree
        set_lower_split_index: Set the index to the node in the next level < split value
        set_upper_split_index: Set the index to the node in the next level > split value
        get_x_data: Get the x data specific to this node
        get_y_data: Get the y data specific to this node

        Private
        -------

        """

        def __init__(self, leaf, x_data, y_data, leaf_type=None):
            """
            Initialize the node.

            :param leaf: The true/false value (defaults to false) to say if a node is a leaf
            :param x_data: The data to be placed into the node
            :param y_data: The y_data to be averaged over if a leaf node
            :param leaf_type: Either classification or regression
            """
            # Set leaf value
            self.__leaf = leaf
            self.__x_data = x_data
            self.__y_data = y_data

            # If a leaf, take average of y_data
            if self.__leaf and (leaf_type == 'regression'):
                self.__prediction = np.mean(y_data)
            elif self.__leaf and (leaf_type == 'classification'):
                temp_counter = Counter(y_data)
                self.__prediction = temp_counter.most_common(1)[0][0]
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

        def set_split(self, idx, val):
            """
            Set the column/split index this node splits on.  Also
            sets the split value for a non-leaf node.

            :param idx: The index
            :param val: Specific value
            """
            self.__col = idx
            self.__split = val

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


class RegressionDecisionTree(DecisionTree):
    """
    Regression Decision tree class.  Will inherit the decision tree class.
    """

    def __init__(self, split_type='rss', leaf_terminate=1):
        """
        Initialize the decision tree superclass.

        :param leaf_terminate: the amount of collections needed to terminate the tree with a leaf (defaults to 1)
        :param split_type: the criteria to split on
        """
        super().__init__('regression', split_type, terminate='leaf', leaf_terminate=leaf_terminate)


class ClassificationDecisionTree(DecisionTree):
    """
    Classification Decision tree class.  Will inherit the decision tree class.

    NOTE: If there is a class tie, this module WILL PICK the lowest class.
    """

    def __init__(self, split_type='gini', terminate='leaf', leaf_terminate=1):
        """
        Initialize the decision tree.

        :param leaf_terminate: the amount of collections needed to terminate the tree with a leaf (defaults to 1)
        :param terminate: the way to terminate the classification tree (leaf/pure)
        :param split_type: the criteria to split on (gini/rss/gain_ratio)
        """
        super().__init__('classification', split_type, terminate=terminate, leaf_terminate=leaf_terminate)
