import numpy as np
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

path = 'src/musicNet/processed_data'
X_train = np.load(path + '/train_data_midi.npy')
X_test = np.load(path + '/test_data_midi.npy')
y_train = np.load(path + '/train_labels_midi.npy')
y_test = np.load(path + '/test_labels_midi.npy')

dt_clf = DecisionTreeClassifier(random_state=42)

dt_clf.fit(X_train, y_train)
training_accuracy = dt_clf.score(X_train, y_train)
accuracy = dt_clf.score(X_test, y_test)
print(training_accuracy)
print(accuracy)

rf_clf = RandomForestClassifier(random_state=42, max_features=512, n_estimators=100)

rf_clf.fit(X_train, y_train)
training_accuracy = rf_clf.score(X_train, y_train)
accuracy = rf_clf.score(X_test, y_test)
print(training_accuracy)
print(accuracy)