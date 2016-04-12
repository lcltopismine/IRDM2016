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
train_data, test_data = get_data()

# filter to one zone and one temp station
zones = 20
tempstns = 11

def selectdata(df, zone, tempstn):
    # filter by zone
    df = df[df.zone_id == zone].copy()

    # pick out correct temp station
    df['temp'] = df['tempstn_'+str(tempstn)]

    # filter to columns of interest
    cols = ['value', 'weight', 'trend', 'temp', 'month', 'dayofweek', 'hour']
    df = df[cols]

    return df

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
extract_and_scale_temps = Pipeline([('extract_temps', ColumnExtractor(cols=[3])),
                                    ('scale_temps', StandardScaler()),
                                    ('poly_temps', PolynomialFeatures(3, include_bias=False))])

# Pipeline to convert Month, Weekday, Hour to one-hot
extract_and_onehot_datetimes = Pipeline([('extract_dt', ColumnExtractor(cols=[4, 5, 6])),
                                         ('onehot_dt', OneHotEncoder())])

# FeatureUnion for basic (unary) features
base_features = FeatureUnion(transformer_list=[('trend', ColumnExtractor(cols=[2])),
                                               ('temps', extract_and_scale_temps),
                                               ('onehots', extract_and_onehot_datetimes)])

# TODO: how to generate the cross-effect terms - month * temp, hour * temp^2 etc.??

# Pipeline
pipe = Pipeline([('base_features', base_features),
                 ('linreg', LinearRegression())])


for zone in range(1, zones+1):
    for tempstn in range(1, tempstns+1):
        train = selectdata(train_data, zone, tempstn)
        test = selectdata(test_data, zone, tempstn)

        X_train = train.values
        X_test = test.values
        y_train = train[['value']]
        y_test = test[['value']]

        pipe.fit(X_train, y_train)
        # y_train_pred = pipe.predict(X_train)
        # y_test_pred = pipe.predict(X_test)

        score_train = pipe.score(X_train, y_train)
        score_test =  pipe.score(X_test, y_test)

        print 'training R2 = %0.5f  test R2 = %0.5f  tempstn = %2i  zone = %2i' % (score_train, score_test, tempstn, zone)
        # print 'evaluate training: %.3f in zone: %i for temperature station: %i' % (mean_squared_error(y_train, y_train_pred)**0.5, zone, tempstn)
        # print 'evaluate test: %.3f in zone: %i for temperature station: %i' % (mean_squared_error(y_test, y_test_pred)**0.5, zone, tempstn)

# NOTE1: this is RMSE not weighted RMSE.  need to incorporate additiona weighting on predictions and total over zones
# NOTE2: this uses test set temperature data to predict for future period.  not sure if this should be available.
# SO - not yet comparable to scores on kaggle.
