import time
import numpy as np
import pandas as pd
import random
import collections
from typing import Any
from types import FunctionType
from types import MethodType
from types import LambdaType

from src.decision_tree.abstract_decision_tree import AbstractDecisionTree
from src.decision_tree.abstract_decision_tree import ConditionNode

class CustomConditionNode(ConditionNode):

    TYPE_CRIT_RANDOM = 0
    TYPE_CRIT_BEST = 1
    
    IMP_FUNC_ENTROPY = 0
    IMP_FUNC_GINI = 1

    def __init__(self, parent: Any              = None,
                       condition: int           = None,
                       children: dict           = {},
                       subset_indeces: set[int] = None,
                       value: int               = None):
        if parent:
            self.df_x: pd.DataFrame = parent.df_x
            self.df_y: pd.DataFrame = parent.df_y
        self.condition: FunctionType       = condition
        self.children: dict[Any, ConditionNode] = children
        self.parent: ConditionNode         = parent
        self.subset_indeces: set[int]      = subset_indeces
        self.value: int                    = self.calculate_value() if value is None else value
        self.dot_attr: collections.defaultdict[str, Any] = collections.defaultdict(str)

    def split(self):
        if self.condition is None:
            raise Exception("Condition is None")
        else:
            df_filtered: pd.DataFrame = self.df_x.loc[list(self.subset_indeces)]
            sx_indices: set = set(df_filtered.loc[df_filtered.apply(self.condition, axis=1)].index.tolist())
            dx_indices: set = set(df_filtered.index.tolist()) - sx_indices        
            sx_node = CustomConditionNode(parent=self, subset_indeces=sx_indices)
            dx_node = CustomConditionNode(parent=self, subset_indeces=dx_indices)
            self.children = {0: sx_node, 1: dx_node}

    def _information_gain(self, attr_series: pd.Series,
                                test_value: float,
                                is_categorical: bool,
                                imp_func: int = IMP_FUNC_ENTROPY) -> float:
        """
        imp_func: 0 = entropy, 1 = gini
        """
        imp_func: MethodType = self._entropy if imp_func == self.IMP_FUNC_ENTROPY else self._gini_impurity
        mask: pd.Series = attr_series == test_value if is_categorical else attr_series <= test_value
        a: int   = sum(mask)
        b: int   = sum(~mask)
        tot: int = a + b
        return imp_func(self.df_y) - (a/tot)*imp_func(self.df_y.loc[list(self.subset_indeces)][mask]) - (b/tot)*imp_func(self.df_y.loc[list(self.subset_indeces)][~mask])
    
    def _generate_attribute_best(self, imp_func=IMP_FUNC_ENTROPY):
        max_info_gain: float         = -1
        max_info_gain_attr_name: str = None
        max_val: float               = None
        max_is_categorical: bool     = False

        for attr_name in self.df_x.columns:
            attr_series: pd.Series = self.df_x[attr_name].loc[list(self.subset_indeces)]
            is_categorical = self._is_categorical(attr_name) 
            
            if is_categorical:
                possible_values: list[int] = attr_series.unique() 
            else:
                n_groups: int = 4
                possible_values: list[float] = attr_series.quantile(np.arange(0, 1, step=1/n_groups) + 1/n_groups).values
            
            local_max_val: float = -1.0
            local_max_ig: float  = -1.0

            for value in possible_values:
                information_gain: float = self._information_gain(attr_series, value, is_categorical, imp_func)
                if information_gain > local_max_ig:
                    local_max_ig = information_gain
                    local_max_val = value
            
            if local_max_ig > max_info_gain:
                max_info_gain           = local_max_ig
                max_val                 = local_max_val
                max_info_gain_attr_name = attr_name
                max_is_categorical      = is_categorical
        
        if self._is_categorical(max_info_gain_attr_name):
            self.condition: LambdaType = lambda row: row[max_info_gain_attr_name] == max_val
        else:
            self.condition: LambdaType = lambda row: row[max_info_gain_attr_name] <= max_val

        print("SPLIT ON", max_info_gain_attr_name, "WITH IG =", max_info_gain)
        
        self.set_dot_attr(max_info_gain_attr_name, max_val, max_info_gain, max_is_categorical)
        
    def _generate_attribute_random(self, imp_func: int = IMP_FUNC_ENTROPY) -> FunctionType:
        """
        imp_func: 0 = entropy, 1 = gini
        """
        index: int             = random.randint(0, len(self.df_x.columns) - 1)
        attr_name: str         = self.df_x.columns[index] # "Payment Currency"
        attr_series: pd.Series = self.df_x[attr_name].loc[list(self.subset_indeces)]
        is_categorical = self._is_categorical(attr_name)

        if is_categorical:
            possible_values: list[int] = attr_series.unique() 
        else:
            n_groups: int = 4 
            possible_values: list[float] = attr_series.quantile(np.arange(0, 1, step=1/n_groups) + 1/n_groups, interpolation="nearest").values
        
        # print(f"Possible values: {possible_values}")

        max_val: float = -1.0
        max_ig: float  = -1.0
        for value in possible_values:
            information_gain: float = self._information_gain(attr_series, value, is_categorical, imp_func)
            if information_gain > max_ig:
                max_ig = information_gain
                max_val = value

        if is_categorical:
            self.condition: LambdaType = lambda row: row[attr_name] == max_val
        else:
            self.condition: LambdaType = lambda row: row[attr_name] <= max_val
       
        print("SPLIT ON", attr_name, "WITH IG =", max_ig)

        self.set_dot_attr(attr_name, max_val, max_ig, is_categorical)

    def generate_condition(self, type_criterion: int = 0, imp_func: int = 0) -> FunctionType:
        """
        type_criterion: 0 = random, 1 = best
        imp_func: 0 = entropy, 1 = gini
        """
        match type_criterion:
            case self.TYPE_CRIT_RANDOM:
                self._generate_attribute_random(imp_func)
            case self.TYPE_CRIT_BEST:
                self._generate_attribute_best(imp_func)
            case _:
                raise ValueError("Invalid type_criterion value")
        return self
    

class CustomDecisionTree(AbstractDecisionTree):
    def __init__(self, criterion, type_criterion=CustomConditionNode.TYPE_CRIT_RANDOM, max_depth=5, min_samples_split=2):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)
        """
        criterion: 0 = entropy, 1 = gini
        type_criterion: 0 = random, 1 = best
        """
        self.criterion: int     = criterion
        self.type_criterion:int = type_criterion
        
        print("PARAMETERS:")
        print("\tCRITERION: " + ("Entropy" if self.criterion == 0 else "Gini"))
        print("\tTYPE CRITERION: " + ("Random" if self.type_criterion == 0 else "Best"))
        print("\tMAX DEPTH: " + str(self.max_depth))
        print("\tMIN SAMPLES SPLIT: " + str(self.min_samples_split))
        print()
    
    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame):
        self.root: ConditionNode = CustomConditionNode(value=round(sum(df_y) / len(df_y)), 
                                                       subset_indeces=set(df_x.index.tolist()))
        self.root.set_df_x(df_x)
        self.root.set_df_y(df_y)

        self.__fit_rec(self.root, 0)

    def __fit_rec(self, node: ConditionNode, depth):
        if (depth >= self.max_depth
            or len(node.subset_indeces) < self.min_samples_split
            or len(set(node.get_labels())) == 1):
            return
        
        node.generate_condition(type_criterion=self.type_criterion, imp_func=self.criterion).split()
        self.__fit_rec(node.children[0], depth + 1)
        self.__fit_rec(node.children[1], depth + 1)

    def __str__(self):
        return super().__str__()
