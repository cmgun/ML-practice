import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import preprocessing
# from sklearn.decomposition import PCA

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

# LabelEncoder
label_encoder = preprocessing.LabelEncoder()
for col in X.columns:
    X[col] = label_encoder.fit_transform(X[col])
Y = label_encoder.fit_transform(Y)

# # PCA
# n_components = 10
# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(X)
# print(X_pca)

model = tree.DecisionTreeClassifier()
model = model.fit(X, Y)

# from sklearn.datasets import load_iris
# iris = load_iris()
# X1, y1 = iris.data, iris.target
# print(iris.feature_names)
# print(iris.target_names)

# visualization
# import graphviz
# dot_data = tree.export_graphviz(model, out_file=None, feature_names=X.columns.tolist(),
#                                 class_names=['edible', 'poisonous'], filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data)

# Loading test data
dataset = pd.read_csv('data/mushroom_for_decisionTree_test.csv', names = head_name)
X_test = dataset.iloc[:, :-1]
Y_test = dataset.iloc[:, -1]
# LabelEncoding
for col in X_test.columns:
    X_test[col] = label_encoder.fit_transform(X_test[col])
Y_test = label_encoder.fit_transform(Y_test)

Y_pre = model.predict(X_test)
a = 0
for i, value in enumerate(Y_pre):
    if Y_test[i] == value:
        a = a + 1
print(a)
print(Y_pre.size)

a = np.sum(Y_pre == Y_test)
b = np.sum(Y_pre != Y_test)
total = Y_pre.size
print(b)