from src.decision_tree.abstract_decision_tree import AbstractDecisionTree
from src.decision_tree.abstract_decision_tree import ConditionNode

import pandas as pd

class DecisionTree(AbstractDecisionTree):
    def __init__(self, criterion, type_criterion=1, max_depth=30, min_samples_split=2):
        super().__init__(criterion, type_criterion, max_depth, min_samples_split)
    
    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame):
        super().fit(df_x, df_y)

        self.root.value = round(sum(df_y) / len(df_y)) 
        self.root.subset_indeces = set(range(len(df_y)))
        # print(self.root.value)
        # print(y.value_counts()[0])
        # print(y.value_counts()[1])
        
        self.__fit_rec(self.root, 0)

    def __fit_rec(self, node: ConditionNode, depth):
        print("__fit_rec")
        if depth >= self.max_depth:
            return
        elif len(node.subset_indeces) < self.min_samples_split:
            return
        elif len(set(self.df_y.take(list(node.subset_indeces)))) == 1:
            return
        
        print("After base cases")
        
        attribute_index, thresholds = self.generate_attribute()
        print("Picked Attribute: " + str(attribute_index))
        

    def __str__(self):
        print("Decision Tree")
        return super().__str__()

