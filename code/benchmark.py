import pandas as pd
from datetime import datetime
import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
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

# preprocessing transformers

# from initial X [dense]
def get_trend_col(X):
    return X[:, 2].reshape((-1,1))

def get_temp_col(X):
    return X[:, 3].reshape((-1,1))

def get_dt_cols(X):
    return X[:, 4:7]

# from 1st round processed X [sparse]
def get_temp_and_dt_onehot_cols(X):
    X=X.tocsc()
    a = X[:, 1]
    b = X[:, 4:]
    return sparse.hstack([a,b])

def get_temp2_and_dt_onehot_cols(X):
    X=X.tocsc()
    a = X[:, 2]
    b = X[:, 4:]
    return sparse.hstack([a,b])

def get_temp3_and_dt_onehot_cols(X):
    X=X.tocsc()
    a = X[:, 3]
    b = X[:, 4:]
    return sparse.hstack([a,b])

def multiply_first_col_by_rest(X):
    if sparse.isspmatrix(X):
        X=X.tocsc()
        first = X[:, 0]
        rest = X[:, 1:]
        res = first.multiply(rest)
        res.tocoo()
    else:
        first = X[:,0].reshape(-1, 1)
        rest = X[:, 1:]
        res = first * rest
    return res

def pass_through(X):
    return X

# Pipeline to scale temperatures
extract_and_scale_temps = Pipeline([('extract_temps', FunctionTransformer(get_temp_col, accept_sparse=True)),
                                    ('scale_temps', StandardScaler()),
                                    ('poly_temps', PolynomialFeatures(3, include_bias=False))])

# Pipeline to convert Month, Weekday, Hour to one-hot
extract_and_onehot_datetimes = Pipeline([('extract_dt', FunctionTransformer(get_dt_cols, accept_sparse=True)),
                                         ('onehot_dt', OneHotEncoder())])

# FeatureUnion for basic (unary) features
base_features = FeatureUnion(transformer_list=[('trend', FunctionTransformer(get_trend_col, accept_sparse=True)),
                                               ('temps', extract_and_scale_temps),
                                               ('onehots', extract_and_onehot_datetimes)])

# Pipeline to create temp * [Month, Weekday, Hour] features
temp_by_datetime_features = Pipeline([('extract_temp_dt', FunctionTransformer(get_temp_and_dt_onehot_cols, accept_sparse=True)),
                                      ('tempbydt', FunctionTransformer(multiply_first_col_by_rest, accept_sparse=True))])
# Pipeline to create temp * [Month, Weekday, Hour] features
temp2_by_datetime_features = Pipeline([('extract_temp_dt', FunctionTransformer(get_temp2_and_dt_onehot_cols, accept_sparse=True)),
                                      ('tempbydt', FunctionTransformer(multiply_first_col_by_rest, accept_sparse=True))])
# Pipeline to create temp * [Month, Weekday, Hour] features
temp3_by_datetime_features = Pipeline([('extract_temp_dt', FunctionTransformer(get_temp3_and_dt_onehot_cols, accept_sparse=True)),
                                      ('tempbydt', FunctionTransformer(multiply_first_col_by_rest, accept_sparse=True))])

# FeatureUnion for cross-features
cross_features = FeatureUnion(transformer_list=[('passthrough', FunctionTransformer(pass_through, accept_sparse=True)),
                                                ('cross1', temp_by_datetime_features),
                                                ('cross2', temp2_by_datetime_features),
                                                ('cross3', temp3_by_datetime_features)])

# Pipeline
pipe = Pipeline([('base_features', base_features),
                 ('cross_features', cross_features),
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
