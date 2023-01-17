class Node:
    def __init__(self, feature_index=None, considered=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.considered = considered
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value
