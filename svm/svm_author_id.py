#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

from sklearn.svm import SVC
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

linear_svc = SVC(kernel='rbf', C=10000.0)

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
linear_svc.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"


tt0 = time()
pred = linear_svc.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"


#numero de predicoes para chris
chris = 0
for p in pred:
	if( p == 1 ):
		chris += 1

print chris


#print pred[10] # predict element 10th
#print pred[26] 
#print pred[50]

print linear_svc.score(features_test, labels_test)


#########################################################


