#!/usr/bin/python
from time import time
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatterplot and identify them visually
grade_fast = [features_train[ii][0]
              for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1]
              for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0]
              for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1]
              for ii in range(0, len(features_train)) if labels_train[ii] == 1]


# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
################################################################################


# your code here!  name your classifier object clf if you want the
# visualization code (prettyPicture) to show you the decision boundary
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import DecisionTreeClassifier as DTC

clf_list = [RFC(), ABC(), KNC()]
acc_list = []
# RandomForestClassifier
# n_estimators ~10
# criterion = 'entropy'/ 'gini'
# max_features ~0.4
# Max acc: 0.94

# KNeighborsClassifier
# n_neighbors: 8
# weights: uniform
# algorithm: any
# Max acc: 0.944

# AdaBoostClassifier
# base_estimator:
# n_estimators: 13-23
# learning_rate: 2.
# Max acc: 0.936
import numpy as np
for base_estimator in [DTC(max_depth=None, min_samples_split=10)]:
    for n_estimators in range(13, 24):
        for learning_rate in [1.18, 1., 2., 1.85]:
            # print '\nn_estimators: %d' % i
            clf = ABC(base_estimator=base_estimator, n_estimators=n_estimators,
                      learning_rate=learning_rate)
            t0 = time()
            clf.fit(features_train, labels_train)
            # print '[%s] Training time: %.3fs' % (clf.__class__.__name__, time() - t0)
            acc = clf.score(features_test, labels_test)
            acc_list.append(
                [acc, base_estimator.__class__.__name__, n_estimators, learning_rate])
            # print '[%s] Accuracy: %.2f' % (clf.__class__.__name__, 100.0 * acc)
acc_list.sort(key=lambda a: (a[0]), reverse=True)
print acc_list[:10]
# for clf in clf_list:
#     t0 = time()
#     clf.fit(features_train, labels_train)
#     print '[%s] Training time: %.3fs' % (clf.__class__.__name__, time() - t0)
#     acc = clf.score(features_test, labels_test)
#     print '[%s] Accuracy: %.2f' % (clf.__class__.__name__, 100.0 * acc)
#     # try:
#     prettyPicture(clf, features_test, labels_test,
#                   clf.__class__.__name__+'.png')
