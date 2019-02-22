#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Fetch MNIST data set
# Note: fetch_mldata is not available in v. > 0.20, use fetch_openml instead
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)

def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]
    
mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
sort_by_target(mnist) # fetch_openml() returns an unsorted dataset


# In[16]:


mnist


# In[17]:


X, y = mnist['data'], mnist['target']


# In[18]:


# 70000 images each 28x28 pixels
X.shape


# In[19]:


y.shape


# In[20]:


# Peek at some of the images
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
          interpolation='nearest')
plt.axis('off')
plt.show()


# In[21]:


y[36000]


# In[22]:


# Create test and train sets

import numpy as np

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[23]:


# Create a binary classifier for number 5

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


# In[24]:


# Use Stochastic Gradient Descent classifier
# This classifier can handle very large datasets
# Deals with training instances independently

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


# In[25]:


sgd_clf.predict([some_digit])


# In[26]:


# EVALUATING PERFORMANCE
# 1. Implement cross validation

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct/len(y_pred))


# In[27]:


from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')


# In[28]:


# 2. Implement Confusion Matrix
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


# In[30]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)


# In[32]:


# 3. Precision and recall

from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)


# In[33]:


recall_score(y_train_5, y_train_pred)


# In[36]:


# Implement harmonic mean
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


# In[37]:


y_scores = sgd_clf.decision_function([some_digit])
y_scores


# In[38]:


threshold = 0
y_some_digit_pred = (y_scores > threshold)


# In[39]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')


# In[40]:


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[41]:


# Plot precision and recall as a function of the threshold

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recalls')
    plt.xlabel('Threshold')
    plt.legend(loc='center left')
    plt.ylim([0,1])


# In[42]:


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# In[43]:


y_train_pred_90 = (y_scores > 70000)
precision_score(y_train_5, y_train_pred_90)


# In[44]:


recall_score(y_train_5, y_train_pred_90)


# In[45]:


# Uitilize ROC curve to analyze precision/recall rate

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# In[46]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
plot_roc_curve(fpr, tpr)
plt.show()


# In[47]:


# Use measurment of area under the curve
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)


# In[50]:


# Train a RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')


# In[51]:


# Use positive class's probability as scores
y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# In[52]:


plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc='lower right')
plt.show()


# In[53]:


roc_auc_score(y_train_5, y_scores_forest)


# In[54]:


# MULTICLASS CLASSIFICATION

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])


# In[55]:


some_digit_scores = sgd_clf.decision_function([some_digit])


# In[56]:


some_digit_scores


# In[59]:


# Use One vs One method
from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)


# In[60]:


forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])


# In[61]:


# Evaluate classifiers
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')


# In[62]:


# Scale the inputs to increase accuracy
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')


# In[63]:


# Error Analysis using confusion matrix

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# In[64]:


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# In[65]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums


# In[66]:


np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# In[67]:


# Multilabel classification

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


# In[68]:


knn_clf.predict([some_digit])


# In[71]:


# Remove noise from images 
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


# In[ ]:




