import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import mean_squared_error

from processandmergedata import *

print 'import data and create features'
train, test = get_data()

# Load data
print 'set up X and y'
Xcols = ['tempstn_1', 'tempstn_2', 'tempstn_3', 'tempstn_4', 'tempstn_5', 'tempstn_6',
         'tempstn_7', 'tempstn_8', 'tempstn_9', 'tempstn_10', 'tempstn_11',
         'holiday', 'summer', 'hour', 'dayofweek', 'month']
X = train[Xcols].values
y = train[['value']].values.flatten()
X_test = test[Xcols].values
y_test = test[['value']].values.flatten()

estimators = 500
# Fit regression model
clf = ensemble.GradientBoostingRegressor(n_estimators=estimators, max_depth=4, min_samples_split=1,
                                         learning_rate=0.01, loss='ls')

clf.fit(X, y)

print 'MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y, clf.predict(X))**0.5,
    mean_squared_error(y_test, clf.predict(X_test))**0.5)

# compute test set deviance
test_score = np.zeros((estimators,), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

# Plot training deviance
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(estimators) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(estimators) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# Plot feature importance
# feature_importance = clf.feature_importances_
# # make importances relative to max importance
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + .5
# plt.subplot(1, 2, 2)
# plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, Xcols[sorted_idx])
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
# plt.show()
