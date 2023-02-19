# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder="passthrough")
X = onehotencoder.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm

# 库里面没有b0对应的x0，所以在第一列加上1
X_train = np.append(arr=np.ones((len(X_train), 1)).astype(int), values=X_train, axis=1)
X_attr = [0, 1, 2, 3, 4, 5]
X_opt = X_train[:, X_attr]
X_opt = np.array(X_opt, dtype=float)
model = sm.OLS(y_train, X_opt).fit()
print(model.summary())
max_iter = 50
index = 1
while True:
    if index >= max_iter:
        break
    pvalues = model.pvalues[1:]
    # 返回的pvalues按照xi的顺序，ndarray
    max_pvalue = pvalues.max()
    if max_pvalue > 0.05:
        remove_index = pvalues.argmax()
        # axis=0，按行删除；1，按列删除；None，按行展开，删除第obj-1位置的数，返回一个行矩阵
        X_opt = np.delete(X_opt, remove_index + 1, axis=1)
        model = sm.OLS(y_train, X_opt).fit()
        index = index + 1
    else:
        break
print(model.summary())