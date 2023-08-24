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

            print("----> ", attr_name, is_categorical)
            
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

        print("tot_instances", tot_instances)
        print(attr_series.value_counts())
        
        for value, n_instances in attr_series.value_counts().items():

            mask: pd.Series = attr_series == value
            info_attr += (n_instances/tot_instances) * self._shannon_entropy(self.df_y.loc[list(self.subset_indeces)][mask])
            split_info -= (n_instances/tot_instances) * np.log2(n_instances/tot_instances)
        
        print("info attr", info_attr)

        true_type: type = type(attr_series.iloc[0])

        info_gain = self._shannon_entropy(self.df_y.loc[list(self.subset_indeces)]) - info_attr

        gain_ratio = 0

        if info_gain != 0 and split_info != 0:
            gain_ratio = info_gain / split_info

        print("IG: ", info_gain, "SPLIT INFO",  split_info)

        print("gain ratio", gain_ratio)

        return gain_ratio, \
               lambda row: true_type(row[attr_name]), \
               attr_series.unique()

    def _compute_gain_ratio_numerical(self, attr_series: pd.Series, attr_name: str) -> tuple[float, LambdaType, pd.Series]:
        np.seterr(divide='ignore', invalid='ignore')
        
        len_T = len(attr_series) # |T|

        filtered_df_y = self.df_y.loc[list(self.subset_indeces)]

        info_T = self._shannon_entropy(filtered_df_y)
        
        attr_series.sort_values(inplace=True)

        thresholds = []

        for i in range(1, len(attr_series)-1):
            thresholds.append((attr_series.iloc[i-1] + attr_series.iloc[i]) / 2)

        len_T_1 = 0
        len_T_2 = len_T

        info_T_1_pos, info_T_1_neg, info_T_2_pos, info_T_2_neg = 0, 0, 0, 0

        max_info_gain = np.NINF
        max_info_gain_threshold = None
        max_info_gain_len_T_1 = 0
        max_info_gain_len_T_2 = 0

        for i in range(0, len(attr_series)-1):
            if filtered_df_y.loc[attr_series.index[i]] == 0:
                info_T_2_neg += 1
            else:
                info_T_2_pos += 1
        
        for i, threshold in enumerate(thresholds):

            len_T_1 += 1
            len_T_2 -= 1

            if filtered_df_y.loc[attr_series.index[i]] == 0:
                info_T_1_neg += 1
                info_T_2_neg -= 1
            else:
                info_T_1_pos += 1
                info_T_2_pos -= 1
            
            # if info_T_1_pos == 0 or info_T_1_neg == 0 or info_T_2_pos == 0 or info_T_2_neg == 0:
            #    pass # continue

            eps = 0

            # print("1) NUM:", info_T_1_pos, "     DEN:", len_T_1, "   DIV:", info_T_1_pos / len_T_1)
            # print("2) NUM:", info_T_1_neg, "     DEN:", len_T_1, "   DIV:", info_T_1_neg / len_T_1)
            # print("3) NUM:", info_T_2_pos, "     DEN:", len_T_2, "   DIV:", info_T_2_pos / len_T_2)
            # print("4) NUM:", info_T_2_neg, "     DEN:", len_T_2, "   DIV:", info_T_2_neg / len_T_2)

            info_T_1 = - info_T_1_pos * np.log2((info_T_1_pos + eps) / len_T_1) - info_T_1_neg * np.log2((info_T_1_neg + eps) / len_T_1)
            info_T_2 = - info_T_2_pos * np.log2((info_T_2_pos + eps) / len_T_2) - info_T_2_neg * np.log2((info_T_2_neg + eps) / len_T_2)

            info_X = (info_T_1 + info_T_2) / len_T

            info_gain = info_T - info_X

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_info_gain_threshold = threshold
                max_info_gain_len_T_1 = len_T_1
                max_info_gain_len_T_2 = len_T_2

        condition = lambda row: row[attr_name] <= max_info_gain_threshold

        split_info = - ((max_info_gain_len_T_1 / len_T) * np.log2(max_info_gain_len_T_1 / len_T) + \
                        (max_info_gain_len_T_2 / len_T) * np.log2(max_info_gain_len_T_2 / len_T))

        gain_ratio = max_info_gain / split_info

        return gain_ratio, \
               condition, \
               [max_info_gain_threshold]

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
    def __init__(self, max_depth: int = 20,
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
