from typing import Any
from types import FunctionType
from abc import ABCMeta
from abc import abstractmethod

import pandas as pd

class ConditionNode:
    def __init__(self, condition=None, children=None, parent=None, subset_indeces=None, value=None):
        self.value: int = value
        self.condition: FunctionType = condition
        self.children: list[ConditionNode] = children
        self.parent: ConditionNode = parent
        self.subset_indeces: set[int] = subset_indeces

    def is_leaf(self):
        return len(self.children) == 0

class AbstractDecisionTree(object, metaclass=ABCMeta):
    TYPE_CRIT_BEST = 0
    TYPE_CRIT_RANDOM = 1

    def __init__(self, criterion, type_criterion, max_depth, min_samples_split):
        self.criterion = criterion
        self.type_criterion = type_criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: ConditionNode = ConditionNode()

    @abstractmethod
    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame):
        self.df_x = df_x
        self.df_y = df_y

    def __predict_rec(self, x, node: ConditionNode):
        val = None
        if node.is_leaf():
            val = node.value
        else:
            branch: int = node.condition(x)
            val = self.__predict_rec(x, node.children[branch])
        return val
    
    def predict(self, x): 
        self._predict_rec(x, self.root)
        
    def predict_train(self, x_list):
        for i in range(len(x_list)):
            yield self.__predict(x_list[i])

    def _generate_attribute_best(self):
        # To choice the best feature to split
            # if the variable is in R:
                # threshold based splitting (quartile based splitting)          
            # else:
                # categorical based splitting (for each value of the feature)
            # choose the feature with the smallest/greatest ConditionNode.condition
        return 0

    def _generate_attribute_random(self):
        import random
        attribute_index = random.randint(0, len(self.df_x.columns) - 1)
        print(self.df_x.dtypes, self.df_x.dtypes[attribute_index], self.df_x.dtypes[attribute_index] == float)
        if self.df_x.dtypes[attribute_index] == float:
            pass
    
    def generate_attribute(self):
        if self.type_criterion == self.TYPE_CRIT_BEST:
            return self._generate_attribute_best()
        else:
            return self._generate_attribute_random(), []

    def __str__(self):
        return ""