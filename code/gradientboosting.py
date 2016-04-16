import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import mean_squared_error

from processandmergedata import *
from wrmse import WRMSE

def main():

    print 'import data and create features'
    train, test = get_data()

    # Load data
    print 'set up X and y'
    Xcols = ['tempstn_1', 'tempstn_2', 'tempstn_3', 'tempstn_4', 'tempstn_5', 'tempstn_6',
             'tempstn_7', 'tempstn_8', 'tempstn_9', 'tempstn_10', 'tempstn_11',
             'holiday', 'summer', 'hour', 'dayofweek', 'month', 'trend']
    X = train[Xcols].values
    y = train[['value']].values.flatten()
    X_test = test[Xcols].values
    y_test = test[['value']].values.flatten()

    # Fit regression model
    print 'fit gradient boosting regressor'
    estimators = 500
    clf = ensemble.GradientBoostingRegressor(n_estimators=estimators, max_depth=4, min_samples_split=1,
                                             learning_rate=0.01, loss='ls')

    clf.fit(X, y)

    # Calculate error
    print 'Root Mean Square Error (zone level only), test: %.5f' % mean_squared_error(y_test, clf.predict(X_test))**0.5
    # Calculate weighted error
    zoneresults = test[['datetime', 'zone_id', 'weight', 'value']].copy()
    zoneresults['prediction'] = clf.predict(X_test)
    print 'Weighted Root Mean Square Error (including system level), test: %.5f' % WRMSE(zoneresults)

    # Compute test set deviance
    print 'compute deviance'
    test_score = np.zeros((estimators,), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)

    # Plot training deviance
    plot_deviance(clf, estimators, test_score)

    # Plot feature importance
    print 'measure feature importance'
    plot_feature_importance(clf.feature_importances_, Xcols)


def plot_deviance(classifier, n_estimators, test_score):
    plt.figure()
    plt.title('Deviance')
    plt.plot(np.arange(n_estimators) + 1, classifier.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.show()


def plot_feature_importance(feat_imp, feat_names):
    feat_imp = 100.0 * (feat_imp / feat_imp.max())
    sorted_idx = np.argsort(feat_imp)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure()
    plt.barh(pos, feat_imp[sorted_idx], align='center')
    features = [feat_names[i].replace('dayofweek', 'weekday').replace('tempstn_', 'station') for i in sorted_idx]
    plt.yticks(pos, features)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


if __name__ == "__main__": main()