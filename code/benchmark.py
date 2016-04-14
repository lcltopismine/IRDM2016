import pandas as pd
from datetime import datetime
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from processandmergedata import *


def main():
    print 'import data and create features'
    train_data, test_data = get_data()

    # counts of zones and temperature stations - for looping
    zones = 20
    tempstns = 11

    n_features = 5


    # BUILD MODEL PIPELINE - this replicates Tao Hong's benchmark model
    # http://repository.lib.ncsu.edu/ir/bitstream/1840.16/6457/1/etd.pdf - page 90

    # LAYER 1 - operates on input X (dense matrix)
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

    # FeatureUnion to create layer 1 features: [Trend, Day, Hour, T, T^2, T^3, Month (onehotx12), Hour (onehotx24)]
    base_features = FeatureUnion(transformer_list=[('trend', FunctionTransformer(get_trend_col)),
                                                   ('day', FunctionTransformer(get_day_col)),
                                                   ('hour', FunctionTransformer(get_hour_col)),
                                                   ('temps', extract_and_scale_temps),
                                                   ('month', extract_and_onehot_month),
                                                   ('hour', extract_and_onehot_hour)])

    # LAYER 2 - operates on output of layer 1 (sparse matrix)
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

    # FeatureUnion to create layer 2 features: [Trend, Month (onehotx12), DayxHour (onehotx168) and
    # (T, T^2, T^3), with each temperature term cross-featured by Month and Hour.
    cross_features = FeatureUnion(transformer_list=[('passthrough', FunctionTransformer(pass_through, accept_sparse=True)),
                                                    ('dayhour', extract_and_onehot_dayhour),
                                                    ('cross1', temp_by_mh_features),
                                                    ('cross2', temp2_by_mh_features),
                                                    ('cross3', temp3_by_mh_features)])

    # PIPELINE - applying linear regression on outputs of Layer 2.
    pipe = Pipeline([('base_features', base_features),
                     ('cross_features', cross_features),
                     ('linreg', LinearRegression())])


    # Run each zone/tempstation combo through pipeline and store best results on training data
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

    feature_importance = np.zeros(n_features)

    for zone in best_tempstn_for_zone:

        train = selectdata(train_data, zone, best_tempstn_for_zone[zone])
        X_train = train.iloc[:, 1:].values
        y_train = train[['value']]

        test = selectdata(test_data, zone, best_tempstn_for_zone[zone])
        X_test = test.iloc[:, 1:].values
        y_test = test[['value']]

        F, _ = f_regression(X_train, y_train.values.ravel())
        feature_importance = feature_importance + F

        pipe.fit(X_train, y_train)

        # save predictions
        y_test_pred = pipe.predict(X_test)
        zoneresults.loc[zoneresults.zone_id == zone, 'prediction'] = y_test_pred

        score_test = pipe.score(X_test, y_test)

        print 'zone = %2i  tempstn = %2i  test R2 = %0.5f' % (zone, tempstn, score_test)

    # calculate errors - system results are just sum over all zones.
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

    # measure feature importance
    print 'Measuring feature importance'
    feature_importance = feature_importance / (zones*tempstns)
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure()
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    features = [train.iloc[:, 1:].columns.values[i] for i in sorted_idx]
    plt.yticks(pos, features)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

# extract required data from dataframe
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

# the below expect dense input - operate on input data
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

# the below expect sparse input - operate on output of earlier layer
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


if __name__ == "__main__": main()