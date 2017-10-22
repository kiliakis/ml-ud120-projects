#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# features_list = ['poi', 'salary']  # You will need to use more features
'''
The num of features is not that much (21), I will start with all of them
and then use pca to limit them to ~10
'''

all_features = ['bonus', 'deferral_payments',
                'deferred_income', 'director_fees', 'email_address',
                'exercised_stock_options', 'expenses', 'from_messages',
                'from_poi_to_this_person', 'from_this_person_to_poi',
                'loan_advances', 'long_term_incentinve', 'other',
                'poi', 'restricted_stock', 'restricted_stock_deferred',
                'salary', 'shared_receipt_with_poi', 'to_messages',
                'total_payments', 'total_stock_value']

features_list = all_features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# sys.exit()
# Task 2: Remove outliers
'''
I will use the outlier cleaner to clean up a bit the data_dict,
perhaps it requires diffent action per feature
'''
# Task 3: Create new feature(s)
'''
I need to create at least 1-2 more features
feature1 = 
feature2 = 
It would be interesting to check on the text of all mails and 
use the invert frequency to add new features, perhaps there are some words
present in the emails that denote a POI
'''
# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
'''
Try 3-4 classifiers, adaboost, randomforest, knearest etc
+ PCA to reduce the num of features to ~10 (without the email text data)
'''

# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

'''
Use the funtion that automatically tunes the classifiers with 2-3 attributes
each with ~10 different values
'''


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
'''
Use different combinations of test data, train data to find the best combination
'''


# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
