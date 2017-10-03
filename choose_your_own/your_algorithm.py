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
plt.show()
################################################################################


# your code here!  name your classifier object clf if you want the
# visualization code (prettyPicture) to show you the decision boundary
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.neighbors import KNeighborsClassifier as KNC

clf_list = [RFC(), ABC(), KNC()]

for clf in clf_list:
    t0 = time()
    clf.fit(features_train, labels_train)
    print '[%s] Training time: %.3fs' % (clf.__class__.__name__, time() - t0)
    acc = clf.score(features_test, labels_test)
    print '[%s] Accuracy: %.2f' % (clf.__class__.__name__, 100.0 * acc)
    # try:
    prettyPicture(clf, features_test, labels_test,
                  clf.__class__.__name__+'.png')
# except NameError:
#     pass
