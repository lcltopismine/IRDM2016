import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

from processandmergedata import *

print 'import data and create features'
train, test = get_data()

# filter to one zone.
# zone = 1
# train = train[train.zone_id == zone]
# test = test[test.zone_id == zone]

print 'set up X and y'
Xcols = ['tempstn_1', 'tempstn_2', 'tempstn_3', 'tempstn_4', 'tempstn_5', 'tempstn_6',
         'tempstn_7', 'tempstn_8', 'tempstn_9', 'tempstn_10', 'tempstn_11',
         'holiday', 'summer', 'hour', 'dayofweek', 'month']
X = train[Xcols].values
y = train[['value']].values.flatten()
X_test = test[Xcols].values
y_test = test[['value']].values.flatten()

print 'partition training set for validation'
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

print 'set up random forest regressor'
forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)

print 'fit random forest regression'
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_val_pred = forest.predict(X_val)

print 'MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_val, y_val_pred))
