import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from processandmergedata import *

print 'import data and create features'
train, test = get_data()

print 'set up X and y'
Xcols = ['tempstn_1', 'tempstn_2', 'tempstn_3', 'tempstn_4', 'tempstn_5', 'tempstn_6',
         'tempstn_7', 'tempstn_8', 'tempstn_9', 'tempstn_10', 'tempstn_11']
X = train[Xcols].values
y = train[['value']]
X_test = test[Xcols].values
y_test = test[['value']]

print 'apply standard scaling'
sc_X = StandardScaler()
sc_y = StandardScaler()
X_std = sc_X.fit_transform(X)
X_std_test = sc_X.transform(X_test)

print 'train a linear regression'
slr = LinearRegression()
slr.fit(X_std, y)

print 'predict'
y_pred = slr.predict(X_std_test)

print 'evaluate'
RMSE = mean_squared_error(y_test, y_pred)**0.5
print RMSE

# NOTE1: this is RMSE not weighted RMSE.  need to incorporate additiona weighting on predictions and total over zones
# NOTE2: this uses test set temperature data to predict for future period.  not sure if this should be available.
# SO - not yet comparable to scores on kaggle.
