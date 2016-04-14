import pandas as pd
from datetime import datetime
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential

from processandmergedata import *
from wrmse import WRMSE


def main():
    print 'import data and create features'
    train_data, test_data = get_data()

    X_raw = selectdata(train_data)
    X_test_raw = selectdata(test_data)
    y = train_data[['value']].values
    y_test = test_data[['value']].values

    # preprocess data - build in to a feature union

    # Pipeline to convert Zone to one-hot
    extract_and_onehot_zone = Pipeline([('extract_zone', FunctionTransformer(get_zone_col)),
                                        ('onehot_zone', OneHotEncoder())])
    # Pipeline to convert Hour to one-hot
    extract_and_onehot_hour = Pipeline([('extract_hour', FunctionTransformer(get_hour_col)),
                                        ('onehot_hour', OneHotEncoder())])
    # Pipeline to convert Day to one-hot
    extract_and_onehot_day = Pipeline([('extract_day', FunctionTransformer(get_day_col)),
                                       ('onehot_day', OneHotEncoder())])
    # Pipeline to convert Month to one-hot
    extract_and_onehot_month = Pipeline([('extract_month', FunctionTransformer(get_month_col)),
                                         ('onehot_month', OneHotEncoder())])
    # Pipeline to convert Holiday to one-hot
    extract_and_onehot_holiday = Pipeline([('extract_holiday', FunctionTransformer(get_holiday_col)),
                                           ('onehot_holiday', OneHotEncoder())])
    # Pipeline to scale temperatures
    extract_and_scale_temps = Pipeline([('extract_temps', FunctionTransformer(get_temp_cols)),
                                        ('scale_temps', StandardScaler())])

    # FeatureUnion
    features = FeatureUnion(transformer_list=[('zone', extract_and_onehot_zone),
                                              ('hour', extract_and_onehot_hour),
                                              ('day', extract_and_onehot_day),
                                              ('month', extract_and_onehot_month),
                                              ('holiday', extract_and_onehot_holiday),
                                              ('temps', extract_and_scale_temps)])

    # PIPELINE - single element to call feature union.
    pipe = Pipeline([('features', features)])

    # process X data [toarray as keras requires dense inputs]
    X = pipe.fit_transform(X_raw).toarray()  #keras requires dense
    X_test = pipe.transform(X_test_raw).toarray()

    input_dim = X.shape[1]

    # build neural network model
    model = Sequential()

    model.add(Dense(input_dim*2, input_dim=input_dim))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(input_dim*2, input_dim=input_dim))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(38))
    model.add(Dense(1))

    print 'build'
    model.compile(loss='mse', optimizer='adam')

    print 'fit'
    model.fit(X, y, nb_epoch=200, batch_size=100)

    print 'evaluate'
    train_score = model.evaluate(X, y, batch_size=100)
    test_score = model.evaluate(X_test, y_test, batch_size=100)
    print 'Root Mean Square Error (zone level only), train: %.5f' % train_score**0.5
    print 'Root Mean Square Error (zone level only), test: %.5f' % test_score**0.5

    print 'predict'
    y_test_pred = model.predict(X_test, batch_size=100, verbose=1)
    zoneresults = test_data[['datetime', 'zone_id', 'weight', 'value']].copy()
    zoneresults['prediction'] = y_test_pred
    print 'Weighted Root Mean Square Error (including system level), test: %.5f' % WRMSE(zoneresults)

    print 'save results'
    save_data_csv(zoneresults, outputfoldername+'nn_results.csv')


def selectdata(df):
    # filter to columns of interest [to break dependency on column ordering provided by processing]
    cols = ['zone_id', 'hour', 'dayofweek', 'month',  'holiday',
            'tempstn_1', 'tempstn_2', 'tempstn_3', 'tempstn_4', 'tempstn_5', 'tempstn_6',
            'tempstn_7', 'tempstn_8', 'tempstn_9', 'tempstn_10', 'tempstn_11']
    df = df[cols].copy()

    return df


# preprocessing transformers

# the below expect dense input - operate on input data
def get_zone_col(X):
    return X[:, 0].reshape((-1, 1))
def get_hour_col(X):
    return X[:, 1].reshape((-1, 1))
def get_day_col(X):
    return X[:, 2].reshape((-1, 1))
def get_month_col(X):
    return X[:, 3].reshape((-1, 1))
def get_holiday_col(X):
    return X[:, 4].reshape((-1, 1))
def get_temp_cols(X):
    return X[:, 5:16]



if __name__ == "__main__": main()