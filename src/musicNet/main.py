import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb

path = 'src/musicNet/processed_data'
X_train = np.load(path + '/train_data_midi.npy')
X_test = np.load(path + '/test_data_midi.npy')
y_train = np.load(path + '/train_labels_midi.npy')
y_test = np.load(path + '/test_labels_midi.npy')

# 0 -> 0 - Bach
# 1 -> 1 - Beethoven
# 2 -> 2 - Brahms
# 7 -> 3 - Mozart
# 9 -> 4 - Schubert

y_train[y_train==7] = 3
y_test[y_test==7] = 3
y_train[y_train==9] = 4
y_test[y_test==9] = 4

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

bst = xgb.XGBClassifier(n_estimators=20, max_depth=15, learning_rate=0.8, objective='multi:softmax')
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
training_accuracy = bst.score(X_train, y_train)
test_accuracy = bst.score(X_test, y_test)
print(training_accuracy)
print(test_accuracy)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

dtrain.save_binary('src/musicNet/data/xgboost/train.buffer')

param = {'max_depth': 10, 'eta': 1, 'objective': 'multi:softmax'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['num_class'] = 5
param['eval_metric'] = ['auc', 'ams@0']

evallist = [(dtrain, 'train'), (dtest, 'eval')]

num_round = 10000
bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=10)

bst.save_model('src\\musicNet\\saved_models\\bt\\austin1.model')
# dump model
bst.dump_model('src\\musicNet\\saved_models\\bt\\dump.raw.txt')
# dump model with feature map
#bst.dump_model('src/musicNet/saved_models/bt/dump.raw.txt', 'src/musicNet/saved_models/bt/featmap.txt')
#xgb.plot_importance(bst)
#xgb.plot_tree(bst, num_trees=2)
#xgb.to_graphviz(bst, num_trees=2)

ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

confusion_mat = confusion_matrix(y_test, ypred)
conf_mat_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
conf_mat_display.plot()

plt.show()