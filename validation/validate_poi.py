#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

from sklearn.cross_validation import train_test_split
data = featureFormat(data_dict, features_list)

labels, features = targetFeatureSplit(data)

labels_train, labels_test, features_train, features_test = train_test_split(labels, features, test_size=0.3, random_state=42)


### it's all yours from here forward!  

from time import time
from sklearn.tree import DecisionTreeClassifier as dt
clf = dt()
t0 = time()
clf.fit(features_train, labels_train)
print 'Training time:', round(time()-t0, 3), 's'

#  t0 = time()
from sklearn.metrics import recall_score, precision_score
pred = clf.predict(features_test).tolist()
print 'POIs Predicted in the test data:', pred.count(1)
print 'Num of True Positives:', sum([pred[i]==labels_test[i] for i in
    range(len(pred)) if pred[i]])
print 'Recall score:', recall_score(labels_test, pred)
print 'Precision score:', precision_score(labels_test, pred)
#  print 'Predicting time:', round(time()-t0, 3), 's'
#  
#  
#  from sklearn.metrics import accuracy_score
#  
#  t0 = time()
#  accuracy = accuracy_score(pred, labels_test)
#  print 'Accuracy score time:', round(time()-t0, 3), 's'
accuracy = clf.score(features_test, labels_test)
print 'The accuracy is', accuracy
