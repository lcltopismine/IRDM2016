import pandas as pd
from datetime import datetime
import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion

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
    cols = ['value', 'trend', 'temp', 'month', 'dayofweek', 'hour']
    df = df[cols]

    return df

# preprocessing transformers


# from initial X [dense]
def get_trend_col(X):
    return X[:, 0].reshape((-1, 1))
def get_temp_col(X):
    return X[:, 1].reshape((-1, 1))
def get_month_col(X):
    return X[:, 2].reshape((-1, 1))
def get_day_col(X):
    return X[:, 3].reshape((-1, 1))
def get_hour_col(X):
    return X[:, 4].reshape((-1, 1))

# from 1st round processed X [sparse]
def get_dayhour(X):
    X = X.tocsc()
    day = X[:, 1]
    hour = X[:, 2]
    dayhour = 24 * day + hour
    return dayhour.todense()

def get_temp_and_mh_onehot_cols(X):
    X = X.tocsc()
    a = X[:, 3]
    b = X[:, 6:]
    return sparse.hstack([a, b])

def get_temp2_and_mh_onehot_cols(X):
    X = X.tocsc()
    a = X[:, 4]
    b = X[:, 6:]
    return sparse.hstack([a, b])

def get_temp3_and_mh_onehot_cols(X):
    X = X.tocsc()
    a = X[:, 5]
    b = X[:, 6:]
    return sparse.hstack([a, b])

def multiply_first_col_by_rest(X):
    if sparse.isspmatrix(X):
        X = X.tocsc()
        first = X[:, 0]
        rest = X[:, 1:]
        res = first.multiply(rest)
    else:
        first = X[:, 0].reshape(-1, 1)
        rest = X[:, 1:]
        res = first * rest
    return res

def pass_through(X):
    X = X.tocsc()
    trend = X[:, 0]
    months = X[:, 6:19]
    return X

# LAYER 1

# Pipeline to convert Month to one-hot
extract_and_onehot_month = Pipeline([('extract_month', FunctionTransformer(get_month_col)),
                                     ('onehot_month', OneHotEncoder())])
# Pipeline to convert Hour to one-hot
extract_and_onehot_hour = Pipeline([('extract_hour', FunctionTransformer(get_hour_col)),
                                    ('onehot_hour', OneHotEncoder())])
# Pipeline to scale temperatures
extract_and_scale_temps = Pipeline([('extract_temps', FunctionTransformer(get_temp_col)),
                                    ('scale_temps', StandardScaler()),
                                    ('poly_temps', PolynomialFeatures(3, include_bias=False))])

# FeatureUnion for basic (unary) features: [Trend, Day, Hour, T, T^2, T^3, Monthx12, Hourx24]
base_features = FeatureUnion(transformer_list=[('trend', FunctionTransformer(get_trend_col)),
                                               ('day', FunctionTransformer(get_day_col)),
                                               ('hour', FunctionTransformer(get_hour_col)),
                                               ('temps', extract_and_scale_temps),
                                               ('month', extract_and_onehot_month),
                                               ('hour', extract_and_onehot_hour)])


# LAYER 2 - now operating on sparse X

# Pipeline to concatenate Day-Hour and convert to one-hot
extract_and_onehot_dayhour = Pipeline([('extract_dayhour', FunctionTransformer(get_dayhour, accept_sparse=True)),
                                       ('onehot_dayhour', OneHotEncoder())])
# Pipeline to create month and hour * temp features
temp_by_mh_features = Pipeline([('extract_temp_mh', FunctionTransformer(get_temp_and_mh_onehot_cols, accept_sparse=True)),
                                ('tempbymh', FunctionTransformer(multiply_first_col_by_rest, accept_sparse=True))])
# Pipeline to create temp * [Month, Weekday, Hour] features
temp2_by_mh_features = Pipeline([('extract_temp2_mh', FunctionTransformer(get_temp2_and_mh_onehot_cols, accept_sparse=True)),
                                 ('temp2bymh', FunctionTransformer(multiply_first_col_by_rest, accept_sparse=True))])
# Pipeline to create temp * [Month, Weekday, Hour] features
temp3_by_mh_features = Pipeline([('extract_temp3_mh', FunctionTransformer(get_temp3_and_mh_onehot_cols, accept_sparse=True)),
                                 ('temp3bymh', FunctionTransformer(multiply_first_col_by_rest, accept_sparse=True))])

# FeatureUnion for cross-features
cross_features = FeatureUnion(transformer_list=[('passthrough', FunctionTransformer(pass_through, accept_sparse=True)),
                                                ('dayhour', extract_and_onehot_dayhour),
                                                ('cross1', temp_by_mh_features),
                                                ('cross2', temp2_by_mh_features),
                                                ('cross3', temp3_by_mh_features)])

# Pipeline
pipe = Pipeline([('base_features', base_features),
                 ('cross_features', cross_features),
                 ('linreg', LinearRegression())])


# Run each zone/tempstation combo through pipeline and remember best results on training data

best_tempstn_for_zone = {}
best_R2score_for_zone = {}

for zone in range(1, zones+1):
    best_tempstn_for_zone[zone] = 0.
    best_R2score_for_zone[zone] = -999.
    for tempstn in range(1, tempstns+1):
        train = selectdata(train_data, zone, tempstn)

        X_train = train.iloc[:, 1:].values
        y_train = train[['value']]

        pipe.fit(X_train, y_train)

        score_train = pipe.score(X_train, y_train)

        # record best temperature station for this zone
        if score_train > best_R2score_for_zone[zone]:
            best_tempstn_for_zone[zone] = tempstn
            best_R2score_for_zone[zone] = score_train

        print 'zone = %2i  tempstn = %2i  training R2 = %0.5f' % (zone, tempstn, score_train)


print 'rerunning best models on test data:'

# store results
zoneresults = test_data[['datetime', 'zone_id', 'weight', 'value']].copy()
zoneresults['prediction'] = 0


for zone in best_tempstn_for_zone:
    train = selectdata(train_data, zone, best_tempstn_for_zone[zone])
    test = selectdata(test_data, zone, best_tempstn_for_zone[zone])

    X_train = train.iloc[:, 1:].values
    X_test = test.iloc[:, 1:].values
    y_train = train[['value']]
    y_test = test[['value']]

    pipe.fit(X_train, y_train)

    y_test_pred = pipe.predict(X_test)
    zoneresults.loc[zoneresults.zone_id == zone, 'prediction'] = y_test_pred

    score_test = pipe.score(X_test, y_test)

    print 'zone = %2i  tempstn = %2i  test R2 = %0.5f' % (zone, tempstn, score_test)

# calculate errors
zoneresults['error'] = zoneresults.value - zoneresults.prediction
sysresults = zoneresults.groupby('datetime')[['weight', 'error']].sum()

# calculate square errors
zoneresults['square_error'] = zoneresults.error ** 2
sysresults['square_error'] = sysresults.error ** 2

# apply weights
zoneresults['weighted_square_error'] = zoneresults.weight * zoneresults.square_error
sysresults['weighted_square_error'] = sysresults.weight * sysresults.square_error

# calculate WRMS
total_weighted_square_error = zoneresults.weighted_square_error.sum() + sysresults.weighted_square_error.sum()
total_weights = zoneresults.weight.sum() + sysresults.weight.sum()
WRMS = (total_weighted_square_error / total_weights) ** 0.5

print 'Weighted Root Mean Square Error: %.5f' % WRMS