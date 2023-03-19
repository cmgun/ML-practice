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
import statsmodels.api as sm

# Error Metrics
from sklearn.metrics import mean_squared_error

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
X1.columns = ['date', 'HSI']
X2 = np.log(idx_data2.loc[:, ('close')]).diff(return_period)
X2 = pd.concat([idx_data2['date'], X2], axis='columns')
X2.columns = ['date', 'VHSI']
# inner join by date
idx_X = pd.merge(X2, X1, on='date', how='left')

# data X3, X4, X5: prada-15, prada-30, para-60 lagged dayReturns
X3 = np.log(stk_data.loc[:, ('close')]).shift(-return_period)
X4 = np.log(stk_data.loc[:, ('close')]).shift(-return_period*3)
X5 = np.log(stk_data.loc[:, ('close')]).shift(-return_period*6)
X6 = np.log(stk_data.loc[:, ('close')]).shift(-return_period*12)
lagged_X = pd.concat([stk_data['date'], X3, X4, X5, X6], axis='columns')
lagged_X.columns = ['date', 'PRADA-5', 'PRADA-15', 'PRADA-30', 'PRADA-60']

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

X = pd.merge(idx_X, lagged_X, on='date', how='left')
X = pd.merge(X, X7, on='date', how='left')
X = pd.merge(X, X8, on='date', how='left')
X = pd.merge(X, X9, on='date', how='left')
X = pd.merge(X, X10, on='date', how='left')
X = pd.merge(X, X11, on='date', how='left')
X = pd.merge(X, X12, on='date', how='left')
X = pd.merge(X, X13, on='date', how='left')
X = pd.merge(X, X14, on='date', how='left')
X = X.dropna()

dataset = pd.merge(Y, X, on='date', how='left')
dataset = dataset.dropna()

# print(dataset)

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

# from pandas.plotting import scatter_matrix
# pyplot.figure(figsize=(15,15))
# scatter_matrix(dataset,figsize=(12,12))
# pyplot.show()

# import statsmodels.api as sm
# Y['date'] = pd.to_datetime(Y['date'])
# Y = Y.set_index('date')
# Y = Y.dropna()
# res = sm.tsa.seasonal_decompose(Y, model='additive', period=52)
# fig = res.plot()
# fig.set_figheight(8)
# fig.set_figwidth(15)
# pyplot.show()

Y = dataset.loc[:, ['date', 'PRADA_pred']]
Y['date'] = pd.to_datetime(Y['date'])
Y = Y.set_index('date')
X['date'] = pd.to_datetime(X['date'])
X = X.set_index('date')
X = X.dropna()
X = X.drop(columns=['EmperorWJ', 'ChowTaiFook', 'ChowSangSang', 'OrientalW', 'KingFook'])
# Y = pd.merge(X['date'], Y, on='date', how='left')

# # Feature Selection
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2, f_regression
#
# bestfeatures = SelectKBest(k=7, score_func=f_regression)
# fit = bestfeatures.fit(X, Y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# featureScores.nlargest(10,'Score').set_index('Specs')  #print 10 best features
# print(featureScores)

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))

validation_size = 0.2

train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]

num_folds = 10
scoring = 'neg_mean_squared_error'

names = []
kfold_results = []
test_results = []
train_results = []
for name, model in models:
    names.append(name)

    ## K Fold analysis:
    kfold = KFold(n_splits=num_folds)
    # converted mean square error to positive. The lower the beter
    cv_results = -1 * cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    kfold_results.append(cv_results)

    # Full Training period
    res = model.fit(X_train, Y_train)
    train_result = mean_squared_error(res.predict(X_train), Y_train)
    train_results.append(train_result)

    # Test results
    test_result = mean_squared_error(res.predict(X_test), Y_test)
    test_results.append(test_result)

    msg = "%s: %f (%f) %f %f" % (name, cv_results.mean(), cv_results.std(), train_result, test_result)
    print(msg)

X_train_ARIMA=X_train.loc[:, ['VHSI', 'HSI', 'LukFook', 'Hengdeli', 'ERNESTBOREL']]
X_test_ARIMA=X_test.loc[:, ['VHSI', 'HSI', 'LukFook', 'Hengdeli', 'ERNESTBOREL']]
tr_len = len(X_train_ARIMA)
te_len = len(X_test_ARIMA)
to_len = len (X)

# ARIMA model
modelARIMA = sm.tsa.arima.ARIMA(endog=Y_train, exog=X_train_ARIMA,order=[1,0,0])
model_fit = modelARIMA.fit()

error_Training_ARIMA = mean_squared_error(Y_train, model_fit.fittedvalues)
predicted = model_fit.predict(start = tr_len -1 ,end = to_len -1, exog = X_test_ARIMA)[1:]
error_Test_ARIMA = mean_squared_error(Y_test,predicted)

# # add ARIMA Training and Test error into result and show it
# train_results.append(error_Training_ARIMA)
# test_results.append(error_Test_ARIMA)
# names.append('ARIMA')
#
# # compare algorithms
# fig = pyplot.figure()
#
# ind = np.arange(len(names))  # the x locations for the groups
# width = 0.35  # the width of the bars
#
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.bar(ind - width/2, train_results,  width=width, label='Train Error')
# pyplot.bar(ind + width/2, test_results, width=width, label='Test Error')
# fig.set_size_inches(15,8)
# pyplot.legend()
# ax.set_xticks(ind)
# ax.set_xticklabels(names)
# pyplot.show()