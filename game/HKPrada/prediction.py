import pandas as pd
import numpy as np
from matplotlib import pyplot

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

import seaborn as sns

# weekly returns, using 5 business day period returns.
return_period = 5
# data head
head_name = ['date', 'open', 'high', 'low', 'close', 'volumn', 'amount']

# data Y: PRADA Future Returns
stk_data = pd.read_csv('data/01913.csv', names = head_name)
# Y = stk_data.loc[:, ['date', 'close']]
Y = np.log(stk_data.loc[:, ('close')]).diff(return_period).\
    shift(-return_period)
Y = pd.concat([stk_data['date'], Y], axis='columns')
Y.columns = ['date', 'PRADA_pred']

# data X1, X2: HSI, VHSI, -5 time
idx_data1 = pd.read_csv('data/HSI.csv', names = head_name)
idx_data2 = pd.read_csv('data/VHSI.csv', names = head_name)
X1 = np.log(idx_data1.loc[:, ('close')]).diff(return_period)
X1 = pd.concat([idx_data1['date'], X1], axis='columns')
X1.columns = ['date', 'lnHSI']
X2 = np.log(idx_data2.loc[:, ('close')]).diff(return_period)
X2 = pd.concat([idx_data2['date'], X2], axis='columns')
X2.columns = ['date', 'lnVHSI']
# inner join by date
idx_X = pd.merge(X2, X1, on='date', how='left')

# data X3, X4, X5: prada-15, prada-30, para-60 lagged dayReturns
X3 = np.log(stk_data.loc[:, ('close')]).shift(-return_period)
X4 = np.log(stk_data.loc[:, ('close')]).shift(-return_period*3)
X5 = np.log(stk_data.loc[:, ('close')]).shift(-return_period*6)
X6 = np.log(stk_data.loc[:, ('close')]).shift(-return_period*12)
lagged_X = pd.concat([stk_data['date'], X3, X4, X5, X6], axis='columns')
lagged_X.columns = ['date', 'PRADA-5', 'PRADA-15', 'PRADA-30', 'PRADA-60']

# todo industry component stock
# data X7 ~ X15: relative component stocks
x7_data = pd.read_csv('data/00590.csv', names = head_name)
X7 = np.log(x7_data.loc[:, ['close']]).shift(-return_period)
X7 = pd.concat([x7_data['date'], X7], axis='columns')
X7.columns = ['date', 'LukFook']
x8_data = pd.read_csv('data/00887.csv', names = head_name)
X8 = np.log(x8_data.loc[:, ['close']]).shift(-return_period)
X8 = pd.concat([x8_data['date'], X8], axis='columns')
X8.columns = ['date', 'EmperorWJ']
x9_data = pd.read_csv('data/01929.csv', names = head_name)
X9 = np.log(x9_data.loc[:, ['close']]).shift(-return_period)
X9 = pd.concat([x9_data['date'], X9], axis='columns')
X9.columns = ['date', 'ChowTaiFook']
x10_data = pd.read_csv('data/00116.csv', names = head_name)
X10 = np.log(x10_data.loc[:, ['close']]).shift(-return_period)
X10 = pd.concat([x10_data['date'], X10], axis='columns')
X10.columns = ['date', 'ChowSangSang']
x11_data = pd.read_csv('data/03389.csv', names = head_name)
X11 = np.log(x11_data.loc[:, ['close']]).shift(-return_period)
X11 = pd.concat([x11_data['date'], X11], axis='columns')
X11.columns = ['date', 'Hengdeli']
x12_data = pd.read_csv('data/00280.csv', names = head_name)
X12 = np.log(x12_data.loc[:, ['close']]).shift(-return_period)
X12 = pd.concat([x12_data['date'], X12], axis='columns')
X12.columns = ['date', 'KingFook']
x13_data = pd.read_csv('data/00398.csv', names = head_name)
X13 = np.log(x13_data.loc[:, ['close']]).shift(-return_period)
X13 = pd.concat([x13_data['date'], X13], axis='columns')
X13.columns = ['date', 'OrientalW']
x14_data = pd.read_csv('data/01856.csv', names = head_name)
X14 = np.log(x14_data.loc[:, ['close']]).shift(-return_period)
X14 = pd.concat([x14_data['date'], X14], axis='columns')
X14.columns = ['date', 'ERNESTBOREL']

dataset = pd.merge(Y, idx_X, on='date', how='right')
dataset = pd.merge(dataset, lagged_X, on='date', how='left')
dataset = pd.merge(dataset, X7, on='date', how='left')
dataset = pd.merge(dataset, X8, on='date', how='left')
dataset = pd.merge(dataset, X9, on='date', how='left')
dataset = pd.merge(dataset, X10, on='date', how='left')
dataset = pd.merge(dataset, X11, on='date', how='left')
dataset = pd.merge(dataset, X12, on='date', how='left')
dataset = pd.merge(dataset, X13, on='date', how='left')
dataset = pd.merge(dataset, X14, on='date', how='left')
dataset = dataset.dropna()

print(dataset)

# # Make a histogram of the DataFrame’s columns
# dataset.hist(bins=50, sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
# pyplot.show()
#
# # show the density distribution
# dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=True, legend=True, fontsize=1, figsize=(15,15))
# pyplot.show()
#
# # Compute pairwise correlation of columns, excluding NA/null values
# correlation = dataset.corr()
# pyplot.figure(figsize=(15,15))
# pyplot.title('Correlation Matrix')
# sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

# dataset_1 = dataset.drop(columns=['EmperorWJ', 'LukFook'])
# print(dataset_1)