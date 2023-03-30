import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import preprocessing
import decisionTree as customizedTree
import treePlotter

# Diable the warnings
import warnings
warnings.filterwarnings('ignore')

# Loading data
head_name = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
            'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring',
            'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
            'population', 'habitat', 'classes']
dataset = pd.read_csv('data/mushroom_for_decisionTree_train.csv', names = head_name)
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]
print(dataset.describe())


# for customized tree
labels = head_name[:-1]
# remove the first row which contains labels
train_dataset = dataset[1:]
train_list = train_dataset.values.tolist()

# construct tree
myTree = customizedTree.create_tree(train_list, labels)

dataset = pd.read_csv('data/mushroom_for_decisionTree_test.csv', names = head_name)
labels = head_name[:-1]
test_dataset = dataset[1:]
test_list = test_dataset.values.tolist()

a = 0
for index, row in enumerate(test_list):
    pre_result = customizedTree.classify(myTree, labels, row)
    if pre_result == row[-1]:
        a = a + 1

print(a)
print(len(test_list))
accuracy = a / len(test_list)
print(accuracy)
# visualize
# treePlotter.create_plot(myTree, figsize=(10, 10))

print('end')