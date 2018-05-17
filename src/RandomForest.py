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
import pandas as pd
import DecisionTree


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

    def __init__(self):
        """
        Ini
        """