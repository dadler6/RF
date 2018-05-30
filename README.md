# RF
This is my own implementation of a random forest, purely for learning purposes.  The file folder structure is as follows:

**src**
The code for the random forest implementation

* [DecisionTree.py](https://github.com/dadler6/RF/blob/master/src/DecisionTree.py): Implementation of a decision tree.
* [RandomForest.py](https://github.com/dadler6/RF/blob/master/src/RandomForest.py):: The implementation of a random forest classifier.

**examples**
Examples of running the various files in src.

* [DecisionTreeClassification\_Examples.ipynb](https://github.com/dadler6/RF/blob/master/examples/DecisionTreeClassification_Examples.ipynb): An example of using the iris dataset from sklearn to make a decision tree classifier using the DecisionTree.py code.

**data**
Data for the various test cases I'm developing.

**tests**
Test cases of the src code.

* [test\_DecisionTree.py](https://github.com/dadler6/RF/blob/master/tests/test_DecisionTree.py): Tests for the DecisionTree.py implementation.
* [test\_RandomForest.py](https://github.com/dadler6/RF/blob/master/tests/test_RandomForest.py): Tests for the RandomForest.py implementation.

## Implementation Notes

### DecisionTree.py

The DecisionTree.py includes an implementation of a **RegressionDecisionTree** and a **ClassificationDecisionTree**.  The docstrings for initialization are as follows:

#### ClassificationDecisionTree
```python
class ClassificationDecisionTree(_DecisionTree):
    def __init__(self, split_type='gini', terminate='leaf', leaf_terminate=1, prune=False):
        """
        Initialize the decision tree.

        :param leaf_terminate: the amount of collections needed to terminate the tree with a leaf (defaults to 1)
        :param terminate: the way to terminate the classification tree (leaf/pure)
        :param split_type: the criteria to split on (gini/rss/gain_ratio)
        :param prune: whether we should use pessimistic pruning on the tree
        """
```

#### RegressionDecisionTree
```python
class RegressionDescisionTree(_DecisionTree):

    def __init__(self, split_type='rss', leaf_terminate=1):
        """
        Initialize the decision tree.

        :param split_type: the criterion to split a node (either rss, gini, gain_ratio)
        :param leaf_terminate: the type of decision tree (classification or regression)
        """
```

Once initialized, each tree has two main methods for usage, fit and predict:

```python
def fit(self, x_data, y_data):
    """
    Fit (train) the decision tree using an inputted dataset.

    :param x_data: The dataset to train the decision tree with.
    :param y_data: The result vector we are regressing on.
    """

def predict(self, x_data):
    """
    Predict a class using the dataset given.

    :param x_data: The dataset to predict
    :return: A vector of predictions for each row in X.
    """
```

where "x\_data" is the input data and "y\_data" is the target data. 

### RandomForest.py

The RandomForest.py file includes an implementation of **RandomForest**.  This RandomForest implementation utilizes the **ClassificationDecisionTree** implementation documented above. The docstring for initialization is as follows:

#### RandomForest
```python
class RandomForest(object):

    def __init__(
            self,
            samp_size=0.5,
            num_trees=10,
            num_features=None,
            split_type='gini',
            terminate='leaf',
            leaf_terminate=1,
            oob=False
    ):
    """
    Initialize the RandomForest class.

    :param samp_size: The number of samples to put within each decision tree
    :param num_trees: The number of trees to make
    :param split_type: The criteria to split on
    :param terminate: The termination criteria
    :param leaf_terminate: The number of samples to put into a leaf
    :param oob: Whether to cross-validated using an out-of-bag samp.e
    """
```

Once initialized, the random forest has two main methods for usage, fit and predict:


```python
def fit(self, x_data, y_data):
    """
    Fit (train) a Random Forest model to the data.

    :param x_data: The dataset to train the decision tree with.
    :param y_data: The result vector we are regressing on.
    """

def predict(self, x_data):
    """
    Predict the y (target) for this x_data

    :param x_data: The daata to predict off of
    :return: The predicted target data (y)
    """
```

where "x\_data" is the input data and "y\_data" is the target data. 