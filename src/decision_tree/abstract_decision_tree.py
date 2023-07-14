from typing import Any
from types import FunctionType
from abc import ABCMeta
from abc import abstractmethod

class ConditionNode:
    def __init__(self, condition=None, sx=None, dx=None, parent=None, value=None):
        self.value: int = value
        self.condition: FunctionType = condition
        self.sx: ConditionNode = sx
        self.dx: ConditionNode = dx
        self.parent: ConditionNode = parent

    def is_leaf(self):
        return self.sx is None and self.dx is None

class AbstractDecisionTree(object, metaclass=ABCMeta):
    CRIT_BEST = 0
    CRIT_RANDOM = 1

    def __init__(self, criterion, type_criterion, max_depth, min_samples_split):
        self.criterion = criterion
        self.type_criterion = type_criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: ConditionNode = ConditionNode()

    @abstractmethod
    def fit(self, x, y):
        pass

    def __predict_train(self, x):
        for i in range(len(x)):
            yield self.__predict(x[i])

    def __predict(self, x): 
        self.__predict_rec(x, self.root)
        
    def __predict_rec(self, x, node: ConditionNode):
        val = None
        if node.is_leaf():
            val = node.value
        elif(not node.condition(x)):
            val = self.__predict_rec(x, node.sx)
        else:
            val = self.__predict_rec(x, node.dx)
        return val

    def __str__(self):
        return ""