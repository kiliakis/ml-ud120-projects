#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)
from time import time

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

from sklearn.tree import DecisionTreeClassifier as dt
clf = dt()
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

arr = clf.feature_importances_
print 'Top importance: %lf' % max(arr)
print 'Index: %d' % arr.tolist().index(max(arr))
print 'Word: %s' % vectorizer.get_feature_names()[arr.tolist().index(max(arr))]
print [(arr[i], vectorizer.get_feature_names()[i]) for i in range(len(arr)) if arr[i] > 0.2]