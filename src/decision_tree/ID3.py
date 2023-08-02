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

class ConditionNodeID3(ConditionNode):
    
    def __init__(self, parent: Any              = None,
                       condition: int           = None,
                       children: list           = [],
                       subset_indeces: set[int] = None,
                       value: int               = None,
                       splitted_attr_names: list[str] = [],
                       numerical_attr_groups = 1):
        super().__init__(parent=parent, condition=condition, children=children, subset_indeces=subset_indeces, value=value)

        self.children = {}
        # Contains the names of the attributes already used for split the tree
        self.splitted_attr_names = splitted_attr_names
        self.numerical_attr_groups = numerical_attr_groups
    
    def generate_condition(self) -> FunctionType:
        max_info_gain: float         = -1
        max_info_gain_attr_name: str = None
        max_is_categorical: bool     = False
        max_condition : LambdaType   = None
        
        for attr_name in self.df_x.columns:
            if attr_name in self.splitted_attr_names:
                continue
            
            attr_series: pd.Series = self.df_x[attr_name].loc[list(self.subset_indeces)]
            is_categorical: bool   = self._is_categorical(attr_name)
            info_gain, condition   = None, None

            if is_categorical:
                # print(attr_name, "CATEGORICAL", attr_series.nunique())
                info_gain, condition, condition_value = self._compute_info_gain_categorical(attr_series, attr_name)
            else:
                # print(attr_name, "NUMERICAL", attr_series.nunique())
                info_gain, condition, condition_value = self._compute_info_gain_numerical(attr_series, attr_name)
            
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_info_gain_attr_name = attr_name
                max_is_categorical = is_categorical
                max_condition = condition

        print("SPLIT ON", max_info_gain_attr_name, "WITH IG =", max_info_gain)

        self.condition = max_condition
        self.splitted_attr_names.append(max_info_gain_attr_name)
        self.set_dot_attr(max_info_gain_attr_name, condition_value , float(max_info_gain), max_is_categorical) # TODO: fix the " " in ancestor
        return self
   
    def _compute_info_gain_categorical(self, attr_series: pd.Series, attr_name: str) -> tuple[float, LambdaType, pd.Series]:
        info_attr = 0
        tot_instances = len(attr_series)
        
        for value, n_instances in attr_series.value_counts().items():
            mask: pd.Series = attr_series == value
            info_attr += (n_instances/tot_instances) * self._shannon_entropy(self.df_y.loc[list(self.subset_indeces)][mask])
        
        return self._shannon_entropy(self.df_y) - info_attr, lambda row: row[attr_name], attr_series.unique()

    def _compute_info_gain_numerical(self, attr_series: pd.Series, attr_name: str) -> tuple[float, LambdaType, pd.Series]:
        info_attr: int = 0
        tot_instances = len(attr_series)
        
        n_groups: int = self.numerical_attr_groups if self.numerical_attr_groups <= attr_series.nunique() else attr_series.nunique()

        quantiles_list = attr_series.quantile(np.arange(0, 1, step=1/n_groups) + 1/n_groups, interpolation="nearest")

        for value, n_instances in quantiles_list.items():
            mask: pd.Series = attr_series <= value
            info_attr += (n_instances/tot_instances) * self._shannon_entropy(self.df_y.loc[list(self.subset_indeces)][mask])

        condition = lambda row: math.floor(stats.percentileofscore(quantiles_list, row[attr_name]) * n_groups / 100) # TODO: check if is correct - 1

        return self._shannon_entropy(self.df_y) - info_attr, condition, quantiles_list.unique()
    
    def split(self):
        if self.condition is None:
            raise Exception("Condition is None")
        else:
            df_filtered: pd.DataFrame = self.df_x.loc[list(self.subset_indeces)]

            grouped = df_filtered.groupby(df_filtered.apply(self.condition, axis=1))
            
            children_indices = {key: group.index.tolist() for key, group in grouped}
            for key in children_indices:
                self.children.update({key: ConditionNodeID3(parent=self, 
                                                            subset_indeces=set(children_indices[key]), 
                                                            splitted_attr_names=self.splitted_attr_names,
                                                            numerical_attr_groups=self.numerical_attr_groups)})
            return self
    
    # def split(self):
    #     if self.condition is None:
    #         raise Exception("Condition is None")
    #     else:
    #         df_filtered: pd.DataFrame = self.df_x.loc[list(self.subset_indeces)]

    #         children_indices = {}
    #         keys = df_filtered.apply(self.condition, axis=1)
    #         for index, key in zip(df_filtered.index, keys):
    #             if key not in children_indices:
    #                 children_indices[key] = []
    #             children_indices[key].append(index)
            
    #         for key in children_indices:
    #             self.children.update({key: ConditionNodeID3(parent=self, 
    #                                                         subset_indeces=set(children_indices[key]), 
    #                                                         splitted_attr_names=self.splitted_attr_names,
    #                                                         numerical_attr_groups=self.numerical_attr_groups)})
    #         return self

class DecisionTreeID3(AbstractDecisionTree):
    def __init__(self, max_depth: int = 5, numerical_attr_groups: int = 1):
        super().__init__(max_depth=max_depth)
        self.numerical_attr_groups: int = numerical_attr_groups
        
        print("PARAMETERS:")
        print("\tMAX DEPTH:", self.max_depth)
        print("\tNUMERICAL ATTR GROUPS:", self.numerical_attr_groups)
        print()

    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame):
        self.root = ConditionNodeID3(value=round(sum(df_y) / len(df_y)), 
                                     subset_indeces=set(df_x.index.tolist()), 
                                     splitted_attr_names=[],
                                     numerical_attr_groups=self.numerical_attr_groups)
        self.root.set_df_x(df_x)
        self.root.set_df_y(df_y)
        
        self.__fit_rec(self.root, 0)

    def __fit_rec(self, node: ConditionNodeID3, depth: int):
        if (depth >= self.max_depth
            or len(node.subset_indeces) < self.min_samples_split
            or len(set(node.get_labels())) == 1
            or len(node.splitted_attr_names) == len(node.df_x.columns)):
            return
            
        # print(f"\nThe nodes {node.splitted_attr_names} have been splitted\n")

        node.generate_condition().split()
        for nd in node.children.values():
            self.__fit_rec(nd, depth + 1)
        return
    
    def print_tree(self):
        self._print_tree_rec(self.root, 0)
    
    def _print_tree_rec(self, node: ConditionNodeID3, depth):
        if node is None or depth == 15:
            return

        print(" "*depth, node.dot_attr["attr_name"])
        for child in node.children.values():
            self._print_tree_rec(child, depth + 1)
