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
                       continuous_attr_groups = 1):
        super().__init__(parent=parent, condition=condition, children=children, subset_indeces=subset_indeces, value=value)

        self.children = {}
        # Contains the names of the attributes already used for split the tree
        self.splitted_attr_names = splitted_attr_names
        self.continuous_attr_groups = continuous_attr_groups
    
    def generate_condition(self) -> FunctionType:

        max_info_gain: float = -1
        max_info_gain_attr_name: str = None
        max_is_categorical: bool = False
        max_condition : LambdaType = None

        for attr_name in self.df_x.columns:

            if attr_name in self.splitted_attr_names:
                continue

            attr_series: pd.Series = self.df_x[attr_name]

            is_categorical: bool = self._is_categorical(attr_name)
            info_gain, condition = None, None

            if is_categorical:
                print(attr_name, "CATEGORICAL", attr_series.nunique())
                info_gain, condition = self._compute_info_gain_categorical(attr_name)
            else:
                print(attr_name, "NUMERICAL", attr_series.nunique())
                info_gain, condition = self._compute_info_gain_numerical(attr_name)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_info_gain_attr_name = attr_name
                max_is_categorical = is_categorical
                max_condition = condition
        
        print("SPLIT ON", max_info_gain_attr_name, "with ig=", max_info_gain)

        #print(self.df_x.iloc[list(self.subset_indeces)][max_info_gain_attr_name].unique())

        # attr_values = pd.Series(self.df_x.iloc[list(self.subset_indeces)][max_info_gain_attr_name].unique())

        # self.condition: LambdaType = lambda row: attr_values[attr_values == row[max_info_gain_attr_name]].index[0]

        attr_series: pd.Series = self.df_x[max_info_gain_attr_name]

        self.condition = max_condition

        self.splitted_attr_names.append(max_info_gain_attr_name)

        self.set_dot_attr(max_info_gain_attr_name, " ", float(max_info_gain), max_is_categorical) # TODO: fix the " " in ancestor
        
        return self
    
    def _compute_info_gain_categorical(self, attr_name: str):
        attr_series : pd.Series = self.df_x[attr_name]
        info_attr = 0
        tot_instances = len(attr_series)

        for value, n_instances in attr_series.value_counts().items():
            mask: pd.Series = attr_series == value
            
            info_attr += (n_instances/tot_instances) * self._entropy(self.df_y[mask])

        return self._entropy(self.df_y) - info_attr, lambda row: row[attr_name]

    def _compute_info_gain_numerical(self, attr_name: str):
        attr_series : pd.Series = self.df_x[attr_name]
        info_attr = 0
        tot_instances = len(attr_series)

        n_groups = self.continuous_attr_groups if self.continuous_attr_groups <= attr_series.nunique() else attr_series.nunique()

        quantiles_list = attr_series.quantile(np.arange(0, 1, step=1 / n_groups), interpolation="nearest")

        for value, n_instances in quantiles_list.items():
            mask: pd.Series = attr_series == value
            
            info_attr += (n_instances/tot_instances) * self._entropy(self.df_y[mask])
        
        condition = lambda row: math.floor(stats.percentileofscore(quantiles_list, row[attr_name]) * n_groups / 100) - 1

        return self._entropy(self.df_y) - info_attr, condition
    
    def split(self):

        if self.condition is None:
            raise Exception("Condition is None")
        else:

            df_filtered: pd.DataFrame = self.df_x.iloc[list(self.subset_indeces)]

            children_indices = {}

            for index, row in df_filtered.iterrows():
                key = self.condition(row)
                if key not in children_indices:
                    children_indices.update({key: []})
                
                children_indices.update({key: children_indices[key] + [index]})

            for key in children_indices:
                # print(key)
                self.children.update({key: ConditionNodeID3(parent=self, subset_indeces=set(children_indices[key]), splitted_attr_names=self.splitted_attr_names)})

            """
            for i in range(len(children_indices)):
                print(i, len(children_indices[i]))
                if len(children_indices[i]) == 0:
                    self.children.append(None)
                else:
                    self.children.append(ConditionNodeID3(parent=self, subset_indeces=set(children_indices[i]), splitted_attr_names=self.splitted_attr_names))
            """
        return self


class DecisionTreeID3(AbstractDecisionTree):
    def __init__(self, max_depth=5, continuous_attr_groups=1):
        super().__init__(max_depth=max_depth)
        self.continuous_attr_groups = continuous_attr_groups
    
    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame):

        self.root = ConditionNodeID3(value=round(sum(df_y) / len(df_y)), subset_indeces=set(df_x.index.tolist()), continuous_attr_groups=self.continuous_attr_groups)

        self.root.set_df_x(df_x)
        self.root.set_df_y(df_y)
             
        self._fit_rec(self.root, 0)

    def _fit_rec(self, node: ConditionNodeID3, depth):
        if ((self.max_depth and depth >= self.max_depth)
            or (len(node.subset_indeces) < self.min_samples_split)
            or (len(set(node.get_labels())) == 1)
            or len(set(node.splitted_attr_names))) >= len(set(node.df_x.columns)):
            return

        # print("DEPTH", depth)
        
        labels_sum = sum(node.get_labels())

        # Test if all labels are equal
        if labels_sum == 0 or labels_sum == len(node.get_labels()):
            return
        
        node.generate_condition().split()

        for nd in node.children.values():
            self._fit_rec(nd, depth + 1)

        # if depth == 0:
        #    self.print_tree()
            
        return
    
    def print_tree(self):
        self._print_tree_rec(self.root, 0)
    
    def _print_tree_rec(self, node: ConditionNodeID3, depth):
        if node is None or depth == 15:
            return

        print("  "*depth, node.dot_attr["attr_name"])
        for child in node.children.values():
            self._print_tree_rec(child, depth + 1)