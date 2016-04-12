import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from processandmergedata import *

print 'import data and create features'
train, test = get_data()

# filter to one zone and one temp station
zone = 1
tempstn = 1

def selectdata(df, zone, tempstn):
    # filter by zone
    df = df[df.zone_id == zone].copy()

    # pick out correct temp station
    df['temp'] = df['tempstn_'+str(tempstn)]

    # filter to columns of interest
    cols = ['value', 'trend', 'temp', 'month', 'dayofweek', 'hour']
    df = df[cols]

    return df

train = selectdata(train, zone, tempstn)
test = selectdata(test, zone, tempstn)

X_train = train.values
X_test = test.values
y_train = train[['value']]
y_test = test[['value']]

# preprocessing

class ColumnExtractor(object):

    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self

# Pipeline to scale temperatures
extract_and_scale_temps = Pipeline([('extract_temps', ColumnExtractor(cols=[2])),
                                    ('scale_temps', StandardScaler()),
                                    ('poly_temps', PolynomialFeatures(3, include_bias=False))])

# Pipeline to convert Month, Weekday, Hour to one-hot
extract_and_onehot_datetimes = Pipeline([('extract_dt', ColumnExtractor(cols=[3, 4, 5])),
                                         ('onehot_dt', OneHotEncoder())])

# FeatureUnion for basic (unary) features
base_features = FeatureUnion(transformer_list=[('trend', ColumnExtractor(cols=[1])),
                                               ('temps', extract_and_scale_temps),
                                               ('onehots', extract_and_onehot_datetimes)])

# TODO: how to generate the cross-effect terms - month * temp, hour * temp^2 etc.??

# Pipeline
pipe = Pipeline([('base_features', base_features),
                 ('linreg', LinearRegression())])


pipe.fit(X_train, y_train)
y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

print 'evaluate training: %.3f' % mean_squared_error(y_train, y_train_pred)**0.5
print 'evaluate training: %.3f' % mean_squared_error(y_test, y_test_pred)**0.5

# NOTE1: this is RMSE not weighted RMSE.  need to incorporate additiona weighting on predictions and total over zones
# NOTE2: this uses test set temperature data to predict for future period.  not sure if this should be available.
# SO - not yet comparable to scores on kaggle.
