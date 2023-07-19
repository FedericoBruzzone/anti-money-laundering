import random
import itertools
from typing import Any
from types import FunctionType
from abc import ABCMeta
from abc import abstractmethod

import pandas as pd
import numpy as np

class ConditionNode:
    TYPE_CRIT_RANDOM = 0
    TYPE_CRIT_BEST = 1
    
    IMP_FUNC_ENTROPY = 0
    IMP_FUNC_GINI = 1

    def __init__(self, condition=None, children=None, parent=None, subset_indeces=None, value=None):
        self.value: int = value
        self.condition: FunctionType = condition
        self.children: list[ConditionNode] = children
        self.parent: ConditionNode = parent
        self.subset_indeces: set[int] = subset_indeces
        
    def _gini_impurity(self, y):
        if isinstance(y, pd.Series):
            prob = y.value_counts() / y.shape[0]
            gini = 1 - np.sum(prob ** 2)
            return gini
        else:
            raise Exception("y must be a pandas Series")
    
    def _scaled_entropy(self, y):
        if isinstance(y, pd.Series):
            prob = y.value_counts() / y.shape[0]
            e = 1e-9
            entropy = np.sum(-prob * np.log2(prob+e))
            return entropy
        else:
            raise Exception("y must be a pandas Series")

    def _information_gain(self, attribute_series, test_value, is_categorical, imp_func=IMP_FUNC_ENTROPY):
        imp_func = self._scaled_entropy if imp_func == self.IMP_FUNC_ENTROPY else self._gini_impurity
        mask = attribute_series == test_value if is_categorical else attribute_series <= test_value
        a = sum(mask)
        b = sum(~mask)
        tot = a + b
        return imp_func(self.df_y) - (a/tot)*imp_func(self.df_y[mask]) - (b/tot)*imp_func(self.df_y[~mask])

    def _categorical_values(self, attribute_series: pd.Series):
        categorical_values = []
        for el in range(len(attribute_series.unique())+1):
            for subset in itertools.combinations(attribute_series.unique(), el):
                categorical_values.append(subset)
        return categorical_values[1:-1]
    
    def _generate_attribute_best(self, imp_func=IMP_FUNC_ENTROPY):
        # To choice the best feature to split
            # if the variable is in R:
                # threshold based splitting (quartile based splitting)          
            # else:imp_func
                # categorical based splitting (for each value of the feature)
            # choose the feature with the smallest/greatest ConditionNode.condition
        y = self.df_y.take(list(self.root.subset_indeces))
        gini = self.gini_impurity(y)
        print("Gini: " + f"{gini:.5f}")
        return 0

    def _generate_attribute_random(self, imp_func=IMP_FUNC_ENTROPY):
        attribute_index = random.randint(0, len(self.df_x.columns) - 1)
        # print(self.df_x.dtypes, self.df_x.dtypes[attribute_index], self.df_x.dtypes[attribute_index] == float)

        attribute_name = self.df_x.columns[attribute_index] # "Payment Currency" 
        attribute_series = self.df_x[attribute_name]

        is_categorical = self.df_x.dtypes[attribute_index] == int and attribute_series.nunique() < 20
        if is_categorical:
            print("Categorical")
            possible_values = attribute_series.unique() # self._categorical_values(attribute_series)
        else:
            print("Numerical")
            step = 0.20
            possible_values = attribute_series.quantile(np.arange(0, 1, step=step)).values
        
        print(possible_values)

        # Find best condition that maximize information gain
        best_val = None
        best_ig = -1
        for value in possible_values:
            information_gain = self._information_gain(attribute_series, value, imp_func)
            if information_gain > best_ig:
                best_ig = information_gain
                best_val = value
        print("Best value: {}".format(best_val))
        print("Best information gain: {}".format(best_ig))
        
        if is_categorical:
            return lambda row: row[attribute_name] == best_val
        else:
            return lambda row: row[attribute_name] <= best_val
                    
    def generate_attribute(self, type_criterion=0, imp_func=0):
        if type_criterion == self.TYPE_CRIT_BEST:
            return self._generate_attribute_best(imp_func)
        else:
            return self._generate_attribute_random(imp_func)

    def get_labels(self):
        return self.df_y.take(list(self.subset_indeces))
    
    def set_df_x(self, df_x: pd.DataFrame):
        self.df_x = df_x
    
    def set_df_y(self, df_y: pd.DataFrame):
        self.df_y = df_y

    def is_leaf(self):
        return len(self.children) == 0

class AbstractDecisionTree(object, metaclass=ABCMeta):

    def __init__(self, criterion, type_criterion, max_depth, min_samples_split):
        self.criterion = criterion
        self.type_criterion = type_criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
    @abstractmethod
    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame):
        self.root: ConditionNode = ConditionNode()
        self.root.value = round(sum(df_y) / len(df_y)) 
        self.root.subset_indeces = set(range(len(df_y)))   
        self.root.set_df_x(df_x)
        self.root.set_df_y(df_y)

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

    def __str__(self):
        return ""