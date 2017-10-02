#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

print len(features_train[0])

from sklearn.tree import DecisionTreeClassifier as dt
clf = dt(min_samples_split=40)
t0 = time()
clf.fit(features_train[:], labels_train[:])
print 'Training time:', round(time()-t0, 3), 's'

t0 = time()
pred = clf.predict(features_test).tolist()
print 'Chis Occurrences:', pred.count(1)
print 'Shara Occurrences:', pred.count(0)
# print 'Predictions: ', pred
print 'Predicting time:', round(time()-t0, 3), 's'


from sklearn.metrics import accuracy_score

t0 = time()
accuracy = accuracy_score(pred, labels_test)
print 'Accuracy score time:', round(time()-t0, 3), 's'

# accuracy = clf.score(features_test, labels_test)
print 'The accuracy is:', accuracy 
#########################################################


