import pandas as pd
import numpy as np
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
                       splitted_attr_names: list[str] = []):
        super().__init__(parent=parent, condition=condition, children=children, subset_indeces=subset_indeces, value=value)

        self.children = {}
        # Contains the names of the attributes already used for split the tree
        self.splitted_attr_names = splitted_attr_names
    
    def generate_condition(self) -> FunctionType:

        max_info_gain: float = -1
        max_info_gain_attr_name: str = None

        for attr_name in self.df_x.columns:

            if attr_name in self.splitted_attr_names:
                continue

            attr_series: pd.Series = self.df_x[attr_name]

            is_categorical: bool = self._is_categorical(attr_name)

            if is_categorical:
                print(attr_name, "CATEGORICAL", attr_series.nunique())
                info_gain = self._compute_info_gain_categorical(attr_series)
            else:
                print(attr_name, "NUMERICAL", attr_series.nunique())
                continue
                # info_gain, threshold = self._compute_info_gain_numerical(attr_series)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_info_gain_attr_name = attr_name
        
        # TODO: remove
        if not max_info_gain_attr_name:
            return self
        
        print("SPLIT ON", max_info_gain_attr_name, "with ig=", max_info_gain)

        #print(self.df_x.iloc[list(self.subset_indeces)][max_info_gain_attr_name].unique())

        # attr_values = pd.Series(self.df_x.iloc[list(self.subset_indeces)][max_info_gain_attr_name].unique())

        # self.condition: LambdaType = lambda row: attr_values[attr_values == row[max_info_gain_attr_name]].index[0]
        self.condition: LambdaType = lambda row: row[max_info_gain_attr_name]

        self.splitted_attr_names.append(max_info_gain_attr_name)

        self.set_dot_attr(max_info_gain_attr_name, " ", float(max_info_gain), is_categorical) # TODO: fix the " " in ancestor
        
        return self
    
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
                print(key)
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
    
    def _compute_info_gain_categorical(self, attr_series: pd.Series):
        info_attr = 0
        tot_instances = len(attr_series)

        for value, n_instances in attr_series.value_counts().items():
            mask: pd.Series = attr_series == value
            
            info_attr += (n_instances/tot_instances) * self._entropy(self.df_y[mask])

        return self._entropy(self.df_y) - info_attr

    def _compute_info_gain_numerical(self, attr_series: pd.Series):
        pass
    
    def is_leaf(self) -> bool:
        return super().is_leaf()


class DecisionTreeID3(AbstractDecisionTree):
    def __init__(self):
        super().__init__(None, None) # TODO: remove mandatory arguments in ancestor
    
    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame):

        self.root = ConditionNodeID3(value=round(sum(df_y) / len(df_y)), subset_indeces=set(df_x.index.tolist()))

        self.root.set_df_x(df_x)
        self.root.set_df_y(df_y)
             
        self._fit_rec(self.root, 0)

    def _fit_rec(self, node: ConditionNodeID3, depth):
        if ((self.max_depth and depth >= self.max_depth)
            or (len(node.subset_indeces) < self.min_samples_split)
            or (len(set(node.get_labels())) == 1)
            or len(set(node.splitted_attr_names))) >= len(set(node.df_x.columns)):
            return
        
        # TODO: remove
        end = True
        for e in node.df_x[list(set(node.df_x.columns) - set(node.splitted_attr_names))].dtypes:
            
            if type(e) == np.dtypes.ObjectDType:
                end = False
                break
        if end:
            return

        print("DEPTH", depth)
        
        labels_sum = sum(node.get_labels())
        print("TEST", labels_sum)

        if labels_sum == 0 or labels_sum == len(node.get_labels()):
            print("tutti uguali!")
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