from src.decision_tree.abstract_decision_tree import AbstractDecisionTree
from src.decision_tree.abstract_decision_tree import ConditionNode

import pandas as pd

class DecisionTree(AbstractDecisionTree):
    def __init__(self, criterion, type_criterion=ConditionNode.TYPE_CRIT_RANDOM, max_depth=30, min_samples_split=2):
        super().__init__(criterion, type_criterion, max_depth, min_samples_split)
    
    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame):
        super().fit(df_x, df_y)
             
        self.__fit_rec(self.root, 0)

    def __fit_rec(self, node: ConditionNode, depth):
        if depth >= self.max_depth:
            print("Max depth reached")
            return
        elif len(node.subset_indeces) < self.min_samples_split:
            print("Min samples split reached")
            return
        elif len(set(node.get_labels())) == 1:
            print("All the same")
            return
        
        print("Generate attribute")
        attribute_index = node.generate_attribute()
        print("Picked Attribute: " + str(attribute_index))
        
    def __str__(self):
        return super().__str__()

