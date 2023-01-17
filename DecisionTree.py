import numpy as np
import pandas as pd
from Node import Node

banknote_datadset = pd.read_csv('BankNote_Authentication.csv')

class DecisionTree:
    size = 0
    def __init__(self, min_samples_split=2, max_depth=2, size=0):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, banknote_datatset, current_depth=0):
        features, labels = banknote_datatset[:, :-1], banknote_datatset[:, -1]
        num_of_samples, num_of_features = np.shape(features)
        self.size += 1
        if num_of_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_node_to_split = self.get_best_split(banknote_datatset, num_of_features)

            if (best_node_to_split["info_gain"] > 0):
                left_subtree = self.build_tree(best_node_to_split["data_left"], current_depth + 1)
                right_subtree = self.build_tree(best_node_to_split["data_right"], current_depth + 1)

                return Node(best_node_to_split["feature_index"], best_node_to_split["considered"], left_subtree,
                            right_subtree,
                            best_node_to_split["info_gain"])

        value_of_leaf = self.calculate_leaf_value(labels)
        return Node(value=value_of_leaf)

    def get_best_split(self, my_data, num_of_features):
        node_to_split = {}
        max_gain = -float("inf")
        for featuer_i in range(num_of_features):
            values = my_data[:, featuer_i]
            considered_nodes = np.unique(values)
            for considered in considered_nodes:
                data_left, data_right = self.split(my_data, featuer_i, considered)
                if len(data_left) > 0 and len(data_right) > 0:
                    y, left_y, right_y = my_data[:, -1], data_left[:, -1], data_right[:, -1]
                    curr_gain = self.information_gain(y, left_y, right_y, "gaining")
                    if curr_gain > max_gain:
                        node_to_split["feature_index"] = featuer_i
                        node_to_split["considered"] = considered
                        node_to_split["data_left"] = data_left
                        node_to_split["data_right"] = data_right
                        node_to_split["info_gain"] = curr_gain
                        max_gain = curr_gain
        return node_to_split
    def split(self, my_data, feature_i, considered):
        data_left = np.array([r for r in my_data if r[feature_i] <= considered])
        data_right = np.array([r for r in my_data if r[feature_i] > considered])
        return data_left, data_right

    def information_gain(self, parent, left_child, right_child, mode="entropy"):
        left_weight = len(left_child) / len(parent)
        right_weight = len(right_child) / len(parent)
        if mode == "gaining":
            result = self.gainig_index(parent) - (
                    left_weight * self.gainig_index(left_child) + right_weight * self.gainig_index(right_child))
        else:
            result = self.entropy(parent) - (
                    left_weight * self.entropy(left_child) + right_weight * self.entropy(right_child))
        return result

    def entropy(self, node):
        class_labels = np.unique(node)
        entropy = 0
        for i in class_labels:
            probability = len(node[node == i]) / len(node)
            entropy += -probability * np.log2(probability)
        return entropy

    def gainig_index(self, node):
        class_labels = np.unique(node)
        gaining = 0
        for i in class_labels:
            probability = len(node[node == i]) / len(node)
            gaining += probability ** 2
        return 1 - gaining

    def calculate_leaf_value(self, node):
        node = list(node)
        return max(node, key=node.count)

    def fit(self, X, Y):
        my_data = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(my_data)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.considered:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def print_tree(self, tree=None, space=" "):
        if not tree: tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "<=", tree.considered, "?", tree.info_gain)
            print("%sleft:" % space, end="")
            self.print_tree(tree.left, space + space)
            print("%sright:" % space, end="")
            self.print_tree(tree.right, space + space)

    def print_size(self, tree=None):
        return self.size
