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
from src.decision_tree.entropy_type           import EntropyType
from src.decision_tree.criterion_type         import CriterionType

class CustomConditionNode(ConditionNode):
    def __init__(self, parent: Any              = None,
                       condition: int           = None,
                       children: dict           = {},
                       subset_indeces: set[int] = None,
                       value: int               = None):
        if parent:
            self.df_x: pd.DataFrame = parent.df_x
            self.df_y: pd.DataFrame = parent.df_y
        self.condition: FunctionType            = condition
        self.children: dict[Any, ConditionNode] = children
        self.parent: ConditionNode              = parent
        self.subset_indeces: set[int]           = subset_indeces
        self.value: int                         = self.calculate_value() if value is None else value
        self.dot_attr: collections.defaultdict[str, Any] = collections.defaultdict(str)

    def split(self):
        if self.condition is None:
            raise Exception("Condition is None")
        else:
            df_filtered: pd.DataFrame = self.df_x.loc[list(self.subset_indeces)]
            grouped = df_filtered.groupby(df_filtered.apply(self.condition, axis=1))
            children_indices = {key: group.index.tolist() for key, group in grouped}
            # print("Here4", children_indices)
            self.children = {key: CustomConditionNode(parent=self, subset_indeces=children_indices[key]) for key in children_indices} 

    def _information_gain(self, attr_series: pd.Series,
                                test_value: float,
                                is_categorical: bool,
                                imp_func: int = EntropyType.SHANNON) -> float:
        match imp_func:
            case EntropyType.SHANNON:
                imp_func: FunctionType = self._shannon_entropy
            case EntropyType.GINI:
                imp_func: FunctionType = self._gini_entropy
            case EntropyType.SCALED:
                imp_func: FunctionType = self._scaled_entropy
        
        mask: pd.Series = attr_series == test_value if is_categorical else attr_series <= test_value
        a: int   = sum(mask)
        b: int   = sum(~mask)
        tot: int = a + b
        entropy_mask = imp_func(self.df_y.loc[list(self.subset_indeces)][mask])
        entropy_not_mask = imp_func(self.df_y.loc[list(self.subset_indeces)][~mask])
        return imp_func(self.df_y) - ((a/tot)*entropy_mask + (b/tot)*entropy_not_mask)
   
    def _generate_attribute_best(self, imp_func=EntropyType.SHANNON,
                                       num_thresholds_numerical_attr: int = 2):
        max_info_gain: float         = np.NINF
        max_info_gain_attr_name: str = None
        max_val: float               = None
        max_is_categorical: bool     = False
        
        for attr_name in self.df_x.columns:
            attr_series: pd.Series = self.df_x[attr_name].loc[list(self.subset_indeces)]
            is_categorical = self._is_categorical(attr_name) 

            if is_categorical:
                possible_values: list[int] = attr_series.unique() 
            else:
                n_groups: int = num_thresholds_numerical_attr
                possible_values: list[float] = attr_series.quantile(np.arange(0, 1, step=1/n_groups)).values
                # possible_values: list[float] = attr_series.unique() 
            
            for value in possible_values:
                information_gain: float = self._information_gain(attr_series, value, is_categorical, imp_func)
                
                if information_gain > max_info_gain:
                    max_info_gain           = information_gain
                    max_val                 = value
                    max_info_gain_attr_name = attr_name
                    max_is_categorical      = is_categorical
        if max_is_categorical:
            self.condition: LambdaType = lambda row: 0 if row[max_info_gain_attr_name] == max_val else 1
        else:
            self.condition: LambdaType = lambda row: 0 if row[max_info_gain_attr_name] <= max_val else 1

        print("SPLIT ON", max_info_gain_attr_name, "WITH IG =", max_info_gain)
        
        self.set_dot_attr(max_info_gain_attr_name, max_val, max_info_gain, max_is_categorical)
        
    def _generate_attribute_random(self, imp_func: int = EntropyType.SHANNON, 
                                         num_thresholds_numerical_attr: int = 2):
        index: int             = random.randint(0, len(self.df_x.columns) - 1)
        attr_name: str         = self.df_x.columns[index] # "Payment Currency"
        attr_series: pd.Series = self.df_x[attr_name].loc[list(self.subset_indeces)]
        is_categorical         = self._is_categorical(attr_name)

        if is_categorical:
            possible_values: list[int] = attr_series.unique() 
        else:
            n_groups: int = num_thresholds_numerical_attr 
            possible_values: list[float] = attr_series.quantile(np.arange(0, 1, step=1/n_groups)).values
        
        # print(f"Possible values: {possible_values}")

        max_val: float = np.NINF
        max_ig: float  = np.NINF
        for value in possible_values:
            information_gain: float = self._information_gain(attr_series, value, is_categorical, imp_func)
            if information_gain > max_ig:
                max_ig = information_gain
                max_val = value
        
        if is_categorical:
            self.condition: LambdaType = lambda row: 0 if row[attr_name] == max_val else 1
        else:
            self.condition: LambdaType = lambda row: 0 if row[attr_name] <= max_val else 1     
       
        print("SPLIT ON", attr_name, "WITH IG =", max_ig)

        self.set_dot_attr(attr_name, max_val, max_ig, is_categorical)

    def generate_condition(self, type_criterion: int = CriterionType.RANDOM, 
                                 imp_func: int = EntropyType.SHANNON,
                                 num_thresholds_numerical_attr: int = 2) -> FunctionType:
        match type_criterion:
            case CriterionType.RANDOM:
                self._generate_attribute_random(imp_func, num_thresholds_numerical_attr)
            case CriterionType.BEST:
                self._generate_attribute_best(imp_func, num_thresholds_numerical_attr)
            case _:
                raise ValueError("Invalid type_criterion value")
        return self
    

class CustomDecisionTree(AbstractDecisionTree):
    def __init__(self, criterion                     = EntropyType.SHANNON,
                       type_criterion                = CriterionType.RANDOM,
                       max_depth                     = 10,
                       min_samples_split             = 2,
                       num_thresholds_numerical_attr = 2):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)
        self.criterion: int        = criterion
        self.type_criterion:int    = type_criterion
        self.num_thresholds_numerical_attr: int = num_thresholds_numerical_attr

        criterion_str: str = ""
        match self.criterion:
            case EntropyType.SHANNON:
                criterion_str = "Entropy"
            case EntropyType.GINI:
                criterion_str = "Gini"
            case EntropyType.SCALED:
                criterion_str = "Scaled"

        print("PARAMETERS:")
        print("\tCRITERION: " + criterion_str)
        print("\tTYPE CRITERION: " + ("Random" if self.type_criterion == 0 else "Best"))
        print("\tMAX DEPTH: " + str(self.max_depth))
        print("\tMIN SAMPLES SPLIT: " + str(self.min_samples_split))
        print("\tNUM THRESHOLDS NUMERICAL ATTR: " + str(self.num_thresholds_numerical_attr))
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
        
        node.generate_condition(type_criterion=self.type_criterion, 
                                imp_func=self.criterion, 
                                num_thresholds_numerical_attr=self.num_thresholds_numerical_attr).split()
       
        if len(node.children) <= 1:
            return

        self.__fit_rec(node.children[0], depth + 1)
        self.__fit_rec(node.children[1], depth + 1)

    def __str__(self):
        return super().__str__()
