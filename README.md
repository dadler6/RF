# RF
This is my own implementation of a random forest, purely for learning purposes.  The file folder structure is as follows:

**src**
The code for the random forest implementation

* DecisionTree.py: Implementation of a decision tree.

**examples**
Examples of running the various files in src.

**data**
Data for the various test cases I'm developing.

**tests**
Test cases of the src code.

* test\_DecisionTree.py: Tests for the DecisionTree.py implementation.

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
