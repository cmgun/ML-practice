# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# 如果取单列就会变成array，模型运算需要矩阵，所以用1:2的方式，取出[1,2)的列，不包括2
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
# 多项式x放进线性回归里
model2 = LinearRegression()
model2.fit(X_poly, y)

# Visualising Linear Regression results and Polynomial Regression results
# reduce the spacing of X to make the polynomial image smoother
X_gird = np.arange(min(X), max(X), 0.1)
X_gird = X_gird.reshape(len(X_gird), 1)
plt.scatter(X, y, color='red')
plt.plot(X, model.predict(X), color='blue', label='linear')
plt.plot(X_gird, model2.predict(poly_reg.fit_transform(X_gird)), color='green', label='polynominal')
plt.legend(loc='upper left')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
