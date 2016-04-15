import pandas as pd
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.metrics import mean_squared_error


from processandmergedata import *
from wrmse import WRMSE


def main():
    print 'import data and create features'
    train_data, test_data = get_data()

    # counts of zones and temperature stations - for looping
    zones = 20
    tempstns = 11

    # BUILD MODEL PIPELINE - this replicates Tao Hong's benchmark model
    # http://repository.lib.ncsu.edu/ir/bitstream/1840.16/6457/1/etd.pdf - page 90

    # LAYER 1 FEATURE CREATION - operates on input X (dense matrix)
    extract_and_scale_trend = make_pipeline(FunctionTransformer(get_trend_col), StandardScaler())
    extract_and_onehot_month = make_pipeline(FunctionTransformer(get_month_col), OneHotEncoder())
    extract_and_onehot_hour = make_pipeline(FunctionTransformer(get_hour_col), OneHotEncoder())
    extract_and_scale_temps = make_pipeline(FunctionTransformer(get_temp_col),
                                            StandardScaler(),
                                            PolynomialFeatures(3, include_bias=False))

    # FeatureUnion to create layer 1 features: [Trend, Day, Hour, T, T^2, T^3, Month (onehotx12), Hour (onehotx24)]
    base_features = FeatureUnion(transformer_list=[('trend', extract_and_scale_trend),
                                                   ('day', FunctionTransformer(get_day_col)),
                                                   ('hour', FunctionTransformer(get_hour_col)),
                                                   ('temps', extract_and_scale_temps),
                                                   ('month', extract_and_onehot_month),
                                                   ('hour', extract_and_onehot_hour)])

    # LAYER 2 FEATURE CREATION - operates on output of layer 1 (sparse matrix)
    extract_and_onehot_dayhour = make_pipeline(FunctionTransformer(get_dayhour, accept_sparse=True), OneHotEncoder())
    temp_by_mh_features = make_pipeline(FunctionTransformer(get_temp_and_mh_onehot_cols, accept_sparse=True),
                                        FunctionTransformer(multiply_first_col_by_rest, accept_sparse=True))
    temp2_by_mh_features = make_pipeline(FunctionTransformer(get_temp2_and_mh_onehot_cols, accept_sparse=True),
                                         FunctionTransformer(multiply_first_col_by_rest, accept_sparse=True))
    temp3_by_mh_features = make_pipeline(FunctionTransformer(get_temp3_and_mh_onehot_cols, accept_sparse=True),
                                         FunctionTransformer(multiply_first_col_by_rest, accept_sparse=True))

    # FeatureUnion to create layer 2 features: [Trend, Month (onehotx12), DayxHour (onehotx168) and
    # (T, T^2, T^3), with each temperature term cross-featured by Month(x12) and Hour(x24).
    cross_features = FeatureUnion(transformer_list=[('passthrough', FunctionTransformer(pass_through, accept_sparse=True)),
                                                    ('dayhour', extract_and_onehot_dayhour),
                                                    ('cross1', temp_by_mh_features),
                                                    ('cross2', temp2_by_mh_features),
                                                    ('cross3', temp3_by_mh_features)])

    # PIPELINE - applying linear regression on outputs of Layer 2.
    pipe = Pipeline([('base_features', base_features),
                     ('cross_features', cross_features),
                     ('linreg', LinearRegression())])

    # For each zone record which temp station yielded the best fitted model
    best_tempstn_for_zone = {}
    best_R2score_for_zone = {}

    # set up arrays for aggregating feature importance statistics
    feat_names = ['Trend', 'Month', 'DayxHour', 'TxMonthHour', 'T2xMonthHour', 'T3xMonthHour']
    n_per_feat = np.array([1, 12, 168, 36, 36, 36])
    feat_imp = np.zeros(len(feat_names))

    print 'create models for each combination of zone and temperature station'
    for zone in range(1, zones+1):

        best_tempstn_for_zone[zone] = 0.
        best_R2score_for_zone[zone] = -999.

        for tempstn in range(1, tempstns+1):

            # get data
            train = selectdata(train_data, zone, tempstn)
            X_train = train.iloc[:, 1:].values
            y_train = train[['value']]

            # fit model
            pipe.fit(X_train, y_train)

            # track feature performance, summing coefficients over related cross-variables
            regressor = pipe.named_steps['linreg']
            coef = np.transpose(np.absolute(regressor.coef_))

            feat_imp = feat_imp + [sum(coef[:1])[0], sum(coef[1:13])[0], sum(coef[13:181])[0], sum(coef[181:217])[0],
                                   sum(coef[217:253])[0], sum(coef[253:289])[0]] / n_per_feat

            # score on training data [default score is R-squared - seek to maximise]
            score_train = pipe.score(X_train, y_train)

            # record best temperature station for this zone
            if score_train > best_R2score_for_zone[zone]:
                best_tempstn_for_zone[zone] = tempstn
                best_R2score_for_zone[zone] = score_train

            print 'zone = %2i  tempstn = %2i  training R2 = %0.5f' % (zone, tempstn, score_train)

    # report on feature importance over training submodels
    print 'Plot feature importance (average over training submodels)'
    plot_feature_importance(feat_imp, feat_names)

    print 'rerun best models on test data'

    # store results
    zoneresults = test_data[['datetime', 'zone_id', 'weight', 'value']].copy()
    zoneresults['prediction'] = 0

    feat_imp = np.zeros(len(feat_names))

    for zone in best_tempstn_for_zone:

        # get data, using best temp station for this zone
        train = selectdata(train_data, zone, best_tempstn_for_zone[zone])
        X_train = train.iloc[:, 1:].values
        y_train = train[['value']]

        test = selectdata(test_data, zone, best_tempstn_for_zone[zone])
        X_test = test.iloc[:, 1:].values
        y_test = test[['value']]

        pipe.fit(X_train, y_train)

        # track feature performance, summing coefficients over related cross-variables
        regressor = pipe.named_steps['linreg']
        coef = np.transpose(np.absolute(regressor.coef_))

        feat_imp = feat_imp + [sum(coef[:1])[0], sum(coef[1:13])[0], sum(coef[13:181])[0], sum(coef[181:217])[0],
                               sum(coef[217:253])[0], sum(coef[253:289])[0]] / n_per_feat

        # save predictions to dataframe
        y_test_pred = pipe.predict(X_test)
        zoneresults.loc[zoneresults.zone_id == zone, 'prediction'] = y_test_pred

        score_test = pipe.score(X_test, y_test)

        print 'zone = %2i  tempstn = %2i  test R2 = %0.5f' % (zone, tempstn, score_test)

    # calculate performance [calling wrmse also saves results]
    rmse = mean_squared_error(zoneresults.value, zoneresults.prediction)**0.5
    wrsme = WRMSE(zoneresults, saveresults=True, modelname='benchmark')
    print 'Root Mean Square Error (zone level only), test: %.5f' % rmse
    print 'Weighted Root Mean Square Error (including system level), test: %.5f' % wrsme

    # report on feature importance
    print 'Plot feature importance'
    plot_feature_importance(feat_imp, feat_names)


def plot_feature_importance(feat_imp, feat_names):
    feat_imp = 100.0 * (feat_imp / feat_imp.max())
    sorted_idx = np.argsort(feat_imp)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure()
    plt.barh(pos, feat_imp[sorted_idx], align='center')
    features = [feat_names[i] for i in sorted_idx]
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


# TRANSFORMERS
# the below expect dense input - operate on input data
def get_trend_col(X):   return X[:, 0].reshape((-1, 1))
def get_temp_col(X):    return X[:, 1].reshape((-1, 1))
def get_month_col(X):   return X[:, 2].reshape((-1, 1))
def get_day_col(X):     return X[:, 3].reshape((-1, 1))
def get_hour_col(X):    return X[:, 4].reshape((-1, 1))

# the below expect sparse input - operate on output of earlier layer
def get_dayhour(X):
    X = X.tocsc()
    day = X[:, 1]
    hour = X[:, 2]
    dayhour = 24 * day + hour
    # return as dense (required by subsequent OneHot transformer)
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
    months = X[:, 6:18]
    return sparse.hstack([trend, months])


if __name__ == "__main__": main()