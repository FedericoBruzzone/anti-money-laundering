import numpy as np
import pandas as pd
import collections
from typing import Any
from types import FunctionType
from types import MethodType
from types import LambdaType
from abc import ABCMeta
from abc import abstractmethod

class ConditionNode(object):

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
    
    @abstractmethod
    def generate_condition(self, type_criterion: int = 0, imp_func: int = 0) -> FunctionType: pass
    
    @abstractmethod
    def split(self): pass
    
    
    def _gini_impurity(self, y: pd.Series) -> float:
        if isinstance(y, pd.Series):
            prob: float = y.value_counts() / y.shape[0]
            gini: float = 1 - np.sum(prob ** 2)
            return gini
        else:
            raise Exception("y must be a pandas Series")
    
    def _entropy(self, y: pd.Series) -> float:
        if isinstance(y, pd.Series):
            prob: float    = y.value_counts() / y.shape[0]
            e: float       = 1e-9
            entropy: float = np.sum(-prob * np.log2(prob+e))
            return entropy
        else:
            raise Exception("y must be a pandas Series")
    
    def calculate_value(self):
        if len(self.subset_indeces) == 0:
            return 0

        # value = round(sum(self.df_y.loc[list(self.subset_indeces)]) / len(self.subset_indeces))

        value = round(sum(self.df_y.filter(list(self.subset_indeces))) / len(self.subset_indeces))

        assert len(self.df_y.filter(list(self.subset_indeces))) == len(self.subset_indeces)
        assert value == 0 or value == 1

        return value

    def get_labels(self) -> pd.Series:
        return self.df_y.iloc[list(self.subset_indeces)]
    
    def set_df_x(self, df_x: pd.DataFrame):
        self.df_x = df_x
    
    def set_df_y(self, df_y: pd.DataFrame):
        self.df_y = df_y

    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def set_dot_attr(self, attr_name : str, condition_value, ig : float, is_categorical : bool):
        self.dot_attr = collections.defaultdict(str, {
            "attr_name": attr_name,
            "condition_value": condition_value,
            "ig": ig,
            "is_categorical": is_categorical
        })

    def str_dot(self) -> str:
        s = ""
        if not self.is_leaf():
            s = f"""{self.dot_attr["attr_name"]} {"=" if self.dot_attr["is_categorical"] else "<="} {self.dot_attr["condition_value"]}
IG: {self.dot_attr["ig"]}"""
            
        return f"""Class: {self.calculate_value()}\n{s}"""
    
    def _is_categorical(self, attr_name: str):
        return self.df_x.dtypes[attr_name] == 'object'

class AbstractDecisionTree(object, metaclass=ABCMeta):

    def __init__(self, criterion, type_criterion, max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.type_criterion = type_criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    @abstractmethod
    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame): pass

    def predict(self, x: pd.Series): 
        return self.__predict_rec(x, self.root)
        
    def predict_test(self, X: pd.DataFrame):
        for i in range(len(X)):
            yield self.predict(X.iloc[i])

    def __predict_rec(self, x: pd.Series, node: ConditionNode):
        val = None
        if node.is_leaf():
            val = node.value
        else:
            branch: int = node.condition(x)
            if node.children[branch] is None:
                val = node.value
            else:
                val = self.__predict_rec(x, node.children[branch])
        return val
    
    # VISUAL REPRESENTATION

    def str_dot(self) -> str:
        dot_str: str = "digraph DecisionTree {\n"
        dot_str += "\trankdir=TD;\n"
        dot_str += "\tnode [shape=box];\n"

        def traverse(node: ConditionNode, parent_id: str) -> str:
            node_id: str = str(id(node))
            dot_str: str = f"\t{node_id} [label=\"{node.str_dot()}\"];\n"
            if parent_id != "":
                dot_str += f"\t{parent_id} -> {node_id};\n"

            if not node.is_leaf():
                for child in node.children.values():
                    child_dot_str: str = traverse(child, node_id)
                    dot_str += child_dot_str
            return dot_str

        dot_str += traverse(self.root, "")
        dot_str += "}\n"
        # print(dot_str)
        return dot_str
    
    def create_dot_files(self, filename: str = "tree.dot", generate_png:bool = False, view: bool = False):

        str_dot = self.str_dot()

        with open(filename, "w") as f:
            f.write(str_dot)

        import subprocess
        if generate_png:
            command: str = f"dot -Tpng {filename} -o tree.png"
            subprocess.run(command, shell=True, check=True) 
        if view:
            # command: str = "nohup xdg-open 'tree.png' >/dev/null 2>&1 &"
            # subprocess.run(command, shell=True, check=True) 

            subprocess.run("code tree.png", shell=True, check=True) 

    def __str__(self) -> str:
        return ""
