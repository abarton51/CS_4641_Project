import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
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

labels = ['Bach', 'Beethoven', 'Brahms', 'Mozart', 'Schubert']

dt_clf = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
training_accuracy = dt_clf.score(X_train, y_train)
accuracy = dt_clf.score(X_test, y_test)
print("Decision Tree Classifier")
print(f"Training Accuracy: {training_accuracy}")
print(f"Test Accuracy: {accuracy}")
print(f"Test F1-Score{f1_score(y_test, y_pred, average='weighted')}\n")
print(dt_clf.get_depth())

path = dt_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("Effective alpha")
ax.set_ylabel("Total impurity of leaves")
ax.set_title("Total Impurity vs Effective alpha for training set")
plt.show()
plt.close()

dt_clfs1 = []
for ccp_alpha in ccp_alphas:
    dt_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    dt_clf.fit(X_train, y_train)
    dt_clfs1.append(dt_clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        dt_clfs1[-1].tree_.node_count, ccp_alphas[-1]
    )
)

train_scores1 = [dt_clf.score(X_train, y_train) for dt_clf in dt_clfs1]
test_scores1 = [dt_clf.score(X_test, y_test) for dt_clf in dt_clfs1]

fig, ax = plt.subplots()
ax.set_xlabel("Alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores1, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores1, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()
plt.close()

dt_clf = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_clf.fit(X_train, y_train)
ypred = dt_clf.predict(X_test)
training_accuracy = dt_clf.score(X_train, y_train)
accuracy = dt_clf.score(X_test, y_test)
print("Decision Tree Classifier")
print(f"Training Accuracy: {training_accuracy}")
print(f"Test Accuracy: {accuracy}")
print(f"Test F1-Score: {f1_score(y_test, ypred, average='weighted')}")
ypred_proba = dt_clf.predict_proba(X_test)
print(f"Test 1v1 AUC-Score: {roc_auc_score(y_test, ypred_proba, average='weighted', multi_class='ovo')}")
print(f"Test 1vRest AUC-Score: {roc_auc_score(y_test, ypred_proba, average='weighted', multi_class='ovr')}\n")
print(dt_clf.get_depth())

confusion_mat = confusion_matrix(y_test, ypred)
conf_mat_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=labels)
conf_mat_display.plot()
plt.title("Decision Tree Classifier - Confusion Matrix")
plt.show()
plt.close()


rf_clf = RandomForestClassifier(random_state=42, max_features=512, n_estimators=100)

rf_clf.fit(X_train, y_train)
training_accuracy = rf_clf.score(X_train, y_train)
accuracy = rf_clf.score(X_test, y_test)
ypred = rf_clf.predict(X_test)
print("Random Forest Classifier")
print(f"Training Accuracy: {training_accuracy}")
print(f"Test Accuracy: {accuracy}")
print(f"Test F1-Score{f1_score(y_test, ypred, average='weighted')}")
ypred_proba = rf_clf.predict_proba(X_test)
print(f"Test 1v1 AUC-Score: {roc_auc_score(y_test, ypred_proba, average='weighted', multi_class='ovo')}")
print(f"Test 1vRest AUC-Score: {roc_auc_score(y_test, ypred_proba, average='weighted', multi_class='ovr')}\n")
max_depth = 0
for tree in rf_clf.estimators_:
    if max_depth < tree.get_depth():
        max_depth = tree.get_depth()
print(f"Maximum depth of Random Forest: {max_depth}\n")

confusion_mat = confusion_matrix(y_test, ypred)
conf_mat_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=labels)
conf_mat_display.plot()
plt.title("Random Forest Classifier - Confusion Matrix")
plt.show()
plt.close()

# ------------- XGBoost ----------------
# Traning model 1

bst = xgb.XGBClassifier(n_estimators=20, max_depth=15, learning_rate=0.8, objective='multi:softmax', verbosity=2, subsample=0.25)
# fit model
bst.fit(X_train, y_train, verbose=True)
# make predictions
training_accuracy = bst.score(X_train, y_train)
test_accuracy = bst.score(X_test, y_test)
ypred = bst.predict(X_test)
print("XGBoost Classifier - 20 estimators, max_depth of 15, learning rate of 0.8, softmax objective function.")
print(f"Training Accuracy: {training_accuracy}")
print(f"Test Accuracy: {accuracy}")
print(f"Test F1-Score{f1_score(y_test, ypred, average='weighted')}")
ypred_proba = bst.predict_proba(X_test)
print(f"Test 1v1 AUC-Score: {roc_auc_score(y_test, ypred_proba, average='weighted', multi_class='ovo')}")
print(f"Test 1vRest AUC-Score: {roc_auc_score(y_test, ypred_proba, average='weighted', multi_class='ovr')}\n")

confusion_mat = confusion_matrix(y_test, ypred)
conf_mat_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat,  display_labels=labels)
conf_mat_display.plot()
plt.title("XGBoost Classifier - Model 1 - Confusion Matrix")
plt.show()
plt.close()
# Model 1 but with table of training results

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {'max_depth': 15, 'eta': 0.8, 'objective': 'multi:softmax'}
param['nthread'] = 4
param['num_class'] = 5
param['subsample'] = 0.25
param['eval_metric'] = ['auc', 'merror']
evallist = [(dtrain, 'train'), (dtest, 'eval')]

num_round = 20
bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=20)
bst.save_model('src\\musicNet\\saved_models\\bt\\austin1.model')
bst.dump_model('src\\musicNet\\saved_models\\bt\\dump.raw.txt')

ypred = bst.predict(dtest)
confusion_mat = confusion_matrix(y_test, ypred)
conf_mat_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat,  display_labels=labels)
conf_mat_display.plot()
plt.title("XGBoost Classifier - Model 1 - Confusion Matrix")
plt.show()
plt.close()

# Training model 2

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

dtrain.save_binary('src/musicNet/data/xgboost/train.buffer')

param = {'max_depth': 10, 'eta': 1, 'objective': 'multi:softmax'}
param['nthread'] = 4
param['subsample'] = 0.25
param['num_class'] = 5
param['eval_metric'] = ['auc', 'merror']
evallist = [(dtrain, 'train'), (dtest, 'eval')]

num_round = 10000
bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=100)
bst.save_model('src\\musicNet\\saved_models\\bt\\austin1.model')
bst.dump_model('src\\musicNet\\saved_models\\bt\\dump.raw.txt')

ypred = bst.predict(dtest)
confusion_mat = confusion_matrix(y_test, ypred)
conf_mat_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat,  display_labels=labels)
conf_mat_display.plot()
plt.title("XGBoost Classifier - Model 2 - Confusion Matrix")
plt.show()
plt.close()

# Repackage model 2 so we can make actual predictions

xgb_clf = xgb.XGBClassifier(**param)
xgb_clf._Boster = bst

xgb_clf.fit(X_train, y_train, verbose=True)
# make predictions
training_accuracy = xgb_clf.score(X_train, y_train)
test_accuracy = xgb_clf.score(X_test, y_test)
ypred = xgb_clf.predict(X_test)
print("XGBoost Classifier - 1000 estimators, max_depth of 15, learning rate of 0.8, softmax objective function.")
print(f"Training Accuracy: {training_accuracy}")
print(f"Test Accuracy: {accuracy}")
print(f"Test F1-Score{f1_score(y_test, ypred, average='weighted')}")
ypred_proba = xgb_clf.predict_proba(X_test)
print(f"Test 1v1 AUC-Score: {roc_auc_score(y_test, ypred_proba, average='weighted', multi_class='ovo')}")
print(f"Test 1vRest AUC-Score: {roc_auc_score(y_test, ypred_proba, average='weighted', multi_class='ovr')}\n")

confusion_mat = confusion_matrix(y_test, ypred)
conf_mat_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat,  display_labels=labels)
conf_mat_display.plot()
plt.title("XGBoost Classifier - Model 3 - Confusion Matrix")
plt.show()
plt.close()