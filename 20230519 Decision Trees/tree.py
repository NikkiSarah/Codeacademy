from collections import Counter
import numpy as np
import operator
import pandas as pd
from random import sample

with open("drug_consumption.data", "r") as f:
    attributes = []
    drug = []
    for line in f:
        attributes.append(line.strip().split(",")[1:13])
        drug.append(line.strip().split(",")[28])


# derive the gini impurity
def gini(data):
    impurity = 1
    label_counts = Counter(data)
    
    for label in label_counts:
        label_prob = label_counts[label] / len(data)
        impurity -= label_prob ** 2
    return impurity


# update the information gain function to weighted information gain
def information_gain(starting_labels, split_labels):
    info_gain = gini(starting_labels)
    for sublist in split_labels:
        gini_sublist = gini(sublist) * len(sublist) / len(starting_labels)
        info_gain -= gini_sublist
    return info_gain


# split_data contains the features split into different subsets
# split_labels contains the labels of those features split into different subsets
def split(features, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in features]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(features)):
            if features[i][column] == k:
                new_data_subset.append(features[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets


# function to determine the index of the feature that causes the best split and the information gain
def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain


def find_best_split_subset(dataset, labels, num_features=4):
    best_gain = 0
    best_feature = 0
    features = np.random.choice(len(dataset[0]), num_features, replace=False)
    for feature in features:
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain


# define leaf and decision node classes
class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value


class Internal_Node:
    def __init__(self, feature, branches, value):
        self.feature = feature
        self.branches = branches
        self.value = value


def build_tree(data, labels, value = ""):
    best_feature, best_gain = find_best_split(data, labels)
    if best_gain == 0:
        return Leaf(Counter(labels), value)
    data_subsets, labels_subsets = split(data, labels, best_feature)

    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree(data_subsets[i], labels_subsets[i], data_subsets[i][0][best_feature])
        branches.append(branch)
    return Internal_Node(best_feature, branches, value)


def build_bagged_tree(data, labels, num_features, value = ""):
    best_feature, best_gain = find_best_split_subset(data, labels, num_features)
    if best_gain < 0.00000001:
        return Leaf(Counter(labels), value)
    data_subsets, labels_subsets = split(data, labels, best_feature)

    branches = []
    for i in range(len(data_subsets)):
        branch = build_bagged_tree(data_subsets[i], labels_subsets[i], num_features, data_subsets[i][0][best_feature])
        branches.append(branch)
    return Internal_Node(best_feature, branches, value)


def print_tree(node, spacing = ""):
    feature_dict = {0: "Age",
                    1: "gender",
                    2: "Education",
                    3: "Country of residence",
                    4: "Ethnicity",
                    5: "Neuroticism score",
                    6: "Extraversion score",
                    7: "Openness to experience score",
                    8: "Agreeableness score",
                    9: "Conscientiousness score",
                    10: "Impulsiveness score",
                    11: "Sensation seeing score"}
    # base case
    if isinstance(node, Leaf):
        print(spacing + str(node.labels))
        return
    # print the feature at this node
    print(spacing + "Splitting on " + feature_dict[node.feature])
    
    # call the function recursively on the true branch
    for i in range(len(node.branches)):
        print(spacing + '--> Branch ' + node.branches[i].value + ':')
        print_tree(node.branches[i], spacing + " ")


def classify(datapoint, tree):
    # check if we're at a leaf
    if isinstance(tree, Leaf):
        # get the label with the highest count
        return max(tree.labels.items(), key=operator.itemgetter(1))[0]
    # otherwise find the branch corresponding to the datapoint
    value = datapoint[tree.feature]

    for branch in tree.branches:
        if branch.value == value:
            return classify(datapoint, branch)
