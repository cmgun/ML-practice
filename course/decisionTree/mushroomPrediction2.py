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

# construct tree
myTree = customizedTree.create_tree(train_dataset, labels)

# visualize
treePlotter.createPlot(myTree, figsize=(10, 10))

print('end')