import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



def load_data(path):
    return pd.read_csv(path, index_col=False)

path_x = os.path.dirname(__file__) + "/features_im00002.csv"
path_y = os.path.dirname(__file__) + "/labels_im00002.csv"

X = load_data(path_x)
X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
Y = load_data(path_y)
Y = Y.loc[:, ~Y.columns.str.contains('^Unnamed')]

Y = Y.replace([255.0], 1)
Y = Y.replace([0.0], 0)

dataset = X.join(Y['l1'])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset['l1']):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]


X_train = strat_train_set.drop('l1', axis=1)
Y_train = strat_train_set['l1'].copy()
X_test = strat_test_set.drop('l1', axis=1)
Y_test = strat_test_set['l1'].copy()

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, Y_train)
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, Y_train)
svm_clf = SVC(gamma='auto')
svm_clf.fit(X_train, Y_train)

results = cross_val_score(sgd_clf, X_train, Y_train, cv=3, scoring="accuracy")

print('============SGD=================\ntraining set:\naccuracy:'+ str(results))

y_train_pred = cross_val_predict(sgd_clf, X_train, Y_train, cv=3)
print('mcc:' + str(matthews_corrcoef(Y_train, y_train_pred)))
print('precision:' + str(precision_score(Y_train, y_train_pred)))
print('recall:' + str(recall_score(Y_train, y_train_pred)))
print('f1:' + str(f1_score(Y_train, y_train_pred)))
results1 = cross_val_score(sgd_clf, X_test, Y_test, cv=3, scoring='accuracy')
y_test_pred = cross_val_predict(sgd_clf, X_test, Y_test, cv=3)
print('test set:\naccuracy:'+str(results1))
print('mcc:' + str(matthews_corrcoef(Y_test, y_test_pred)))
print('precision:' + str(precision_score(Y_test, y_test_pred)))
print('recall:' + str(recall_score(Y_test, y_test_pred)))
print('f1:' + str(f1_score(Y_test, y_test_pred)))

print("train_confusion:")
print(confusion_matrix(Y_train, y_train_pred))
print("test_confusion:")
print(confusion_matrix(Y_test, y_test_pred))


results = cross_val_score(knn_clf, X_train, Y_train, cv=3, scoring="accuracy")
print('============KNN=================\ntraining set:\naccuracy:'+ str(results))

y_train_pred = cross_val_predict(knn_clf, X_train, Y_train, cv=3)
print('mcc:' + str(matthews_corrcoef(Y_train, y_train_pred)))
print('precision:' + str(precision_score(Y_train, y_train_pred)))
print('recall:' + str(recall_score(Y_train, y_train_pred)))
print('f1:' + str(f1_score(Y_train, y_train_pred)))
results1 = cross_val_score(knn_clf, X_test, Y_test, cv=3, scoring='accuracy')
y_test_pred = cross_val_predict(knn_clf, X_test, Y_test, cv=3)
print('test set:\naccuracy:'+str(results1))
print('mcc:' + str(matthews_corrcoef(Y_test, y_test_pred)))
print('precision:' + str(precision_score(Y_test, y_test_pred)))
print('recall:' + str(recall_score(Y_test, y_test_pred)))
print('f1:' + str(f1_score(Y_test, y_test_pred)))

print("train_confusion:")
print(confusion_matrix(Y_train, y_train_pred))
print("test_confusion:")
print(confusion_matrix(Y_test, y_test_pred))


results = cross_val_score(svm_clf, X_train, Y_train, cv=3, scoring="accuracy")
print('============SVM=================\ntraining set:\naccuracy:'+ str(results))

y_train_pred = cross_val_predict(svm_clf, X_train, Y_train, cv=3)
print('mcc:' + str(matthews_corrcoef(Y_train, y_train_pred)))
print('precision:' + str(precision_score(Y_train, y_train_pred)))
print('recall:' + str(recall_score(Y_train, y_train_pred)))
print('f1:' + str(f1_score(Y_train, y_train_pred)))
results1 = cross_val_score(svm_clf, X_test, Y_test, cv=3, scoring='accuracy')
y_test_pred = cross_val_predict(svm_clf, X_test, Y_test, cv=3)
print('test set:\naccuracy:'+str(results1))
print('mcc:' + str(matthews_corrcoef(Y_test, y_test_pred)))
print('precision:' + str(precision_score(Y_test, y_test_pred)))
print('recall:' + str(recall_score(Y_test, y_test_pred)))
print('f1:' + str(f1_score(Y_test, y_test_pred)))

print("train_confusion:")
print(confusion_matrix(Y_train, y_train_pred))
print("test_confusion:")
print(confusion_matrix(Y_test, y_test_pred))





