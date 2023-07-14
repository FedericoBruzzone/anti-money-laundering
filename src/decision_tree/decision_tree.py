from src.decision_tree.abstract_decision_tree import AbstractDecisionTree

class DecisionTree(AbstractDecisionTree):
    def __init__(self, criterion, type_criterion=0, max_depth=None, min_samples_split=2):
        super().__init__(criterion, type_criterion, max_depth, min_samples_split)
    
    def fit(self, x, y):
        self.root.value = round(sum(y) / len(y)) 
        # print(self.root.value)
        # print(y.value_counts()[0])
        # print(y.value_counts()[1])
        
        self.__fit_rec(x, y, self.root, 0)

    def __fit_rec(self, x, y, node, depth):
        if depth >= self.max_depth:
            return
        elif len(x) < self.min_samples_split:
            return
        elif len(set(y)) == 1:
            return
        
        # To choice the best feature to split
            # if the variable is in R:
                # threshold based splitting (quartile based splitting)          
            # else:
                # categorical based splitting (for each value of the feature)
            # choose the feature with the smallest/greatest ConditionNode.condition

    def __str__(self):
        print("Decision Tree")
        return super().__str__()

