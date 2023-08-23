import time
import math
import pandas as pd
import numpy as np
from scipy import stats
from typing import Any
from types import FunctionType
from types import MethodType
from types import LambdaType

from src.decision_tree.abstract_decision_tree import AbstractDecisionTree
from src.decision_tree.abstract_decision_tree import ConditionNode

class ConditionNodeC45(ConditionNode):
    
    def __init__(self, parent: Any                    = None,
                       condition: int                 = None,
                       children: list                 = [],
                       subset_indeces: set[int]       = None,
                       value: int                     = None,
                       splitted_attr_names: list[str] = []):
        super().__init__(parent=parent, condition=condition, children=children, subset_indeces=subset_indeces, value=value)

        self.children = {}
        # Contains the names of the attributes already used for split the tree
        self.splitted_attr_names = splitted_attr_names
    
    def generate_condition(self) -> FunctionType:
        max_gain_ratio: float         = np.NINF
        max_gain_ratio_attr_name: str = None
        max_is_categorical: bool     = False
        max_condition : LambdaType   = None
        
        for attr_name in self.df_x.columns:
            if attr_name in self.splitted_attr_names:
                continue
             
            attr_series: pd.Series = self.df_x[attr_name].loc[list(self.subset_indeces)]
            is_categorical: bool   = self._is_categorical(attr_name)
            gain_ratio, condition   = None, None
            
            if is_categorical:
                gain_ratio, condition, condition_value = self._compute_gain_ratio_categorical(attr_series, attr_name)
            else:
                gain_ratio, condition, condition_value = self._compute_gain_ratio_numerical(attr_series, attr_name)
            
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                max_gain_ratio_attr_name = attr_name
                max_is_categorical = is_categorical
                max_condition = condition
            
        print("SPLIT ON", max_gain_ratio_attr_name, "WITH GAIN RATIO =", max_gain_ratio, "IS CATEGORICAL =", max_is_categorical)

        self.condition = max_condition
        self.splitted_attr_names.append(max_gain_ratio_attr_name)
        self.set_attrs(max_gain_ratio_attr_name, 
                       condition_value , 
                       float(max_gain_ratio), 
                       max_is_categorical)
        return self
   
    def _compute_gain_ratio_categorical(self, attr_series: pd.Series, attr_name: str) -> tuple[float, LambdaType, pd.Series]:
        info_attr, split_info = 0, 0
        tot_instances = len(attr_series)
        
        for value, n_instances in attr_series.value_counts().items():
            mask: pd.Series = attr_series == value
            info_attr += (n_instances/tot_instances) * self._shannon_entropy(self.df_y.loc[list(self.subset_indeces)][mask])
            split_info -= (n_instances/tot_instances) * np.log2(n_instances/tot_instances)
        
        true_type: type = type(attr_series.iloc[0])

        info_gain = self._shannon_entropy(self.df_y.loc[list(self.subset_indeces)]) - info_attr
        gain_ratio = info_gain / split_info

        return gain_ratio, \
               lambda row: true_type(row[attr_name]), \
               attr_series.unique()

    def _compute_gain_ratio_numerical(self, attr_series: pd.Series, attr_name: str) -> tuple[float, LambdaType, pd.Series]:
        info_attr, split_info = 0, 0
        tot_instances = len(attr_series)
        
        # TODO

        condition = lambda row: None

        info_gain = self._shannon_entropy(self.df_y.loc[list(self.subset_indeces)]) - info_attr
        gain_ratio = info_gain / split_info

        return self._shannon_entropy(self.df_y.loc[list(self.subset_indeces)]) - info_attr, \
               condition, \
               None # TODO

    def split(self):
        if self.condition is None:
            raise Exception("Condition is None")
        else:
            df_filtered: pd.DataFrame = self.df_x.loc[list(self.subset_indeces)]
            grouped = df_filtered.groupby(df_filtered.apply(self.condition, axis=1))
            children_indices = {key: group.index.tolist() for key, group in grouped}
            for key, group in children_indices.items():
                self.children.update({key: ConditionNodeC45(parent=self, 
                                                            subset_indeces=set(group), 
                                                            splitted_attr_names=self.splitted_attr_names)})
            return self
    
class DecisionTreeC45(AbstractDecisionTree):
    def __init__(self, max_depth: int = 10,
                       min_samples_split = 2,
                       VERBOSE: bool = True):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)
        
        if VERBOSE:
            print("PARAMETERS:")
            print("\tMAX DEPTH:", self.max_depth)
            print()

    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame):
        self.root = ConditionNodeC45(value=round(sum(df_y) / len(df_y)), 
                                     subset_indeces=set(df_x.index.tolist()), 
                                     splitted_attr_names=[])
        self.root.set_df_x(df_x)
        self.root.set_df_y(df_y)
        self.__fit_rec(self.root, 0)

    def __fit_rec(self, node: ConditionNodeC45, depth: int):
        if (depth >= self.max_depth
            or len(node.subset_indeces) < self.min_samples_split
            or len(set(node.get_labels())) == 1
            or len(node.splitted_attr_names) == len(node.df_x.columns)):
            return
            
        # print(f"\nThe nodes {node.splitted_attr_names} have been splitted\n")

        node.generate_condition().split()

        if len(node.children) <= 1:
            return

        for nd in node.children.values():
            self.__fit_rec(nd, depth + 1)
        return
    
    def print_tree(self):
        self._print_tree_rec(self.root, 0)
    
    def _print_tree_rec(self, node: ConditionNodeC45, depth):
        if node is None or depth == 15:
            return

        print(" "*depth, node.attrs["attr_name"])
        for child in node.children.values():
            self._print_tree_rec(child, depth + 1)
