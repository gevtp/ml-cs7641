"""
ML (CS7641) - Supervised Learning
Student Name: Gevorg Ter-Petrosyan
GT User ID: gterpetrosyan3
"""
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split, learning_curve, validation_curve

import matplotlib.pyplot as plt


# In[ ]:


# DIAMONDS data analysis


# In[2]:


# Load dataset

wineDF = pd.read_csv('winequality-white.csv',sep=';')


# In[6]:


print(wineDF)


# In[3]:


# pre-process dataset
# ord_encoder = OrdinalEncoder(categories=[['Fair','Good','Very Good','Premium','Ideal']])
# diamondsDF["cut"] = ord_encoder.fit_transform(diamondsDF[['cut']])
# print(transformed_cut)
#  = transformed_cut
# print(diamondsDF)
# hot_encoder = OneHotEncoder(handle_unknown='ignore')
# diamonds_color_clarity = hot_encoder.fit_transform(diamondsDF[['color','clarity']]).toarray()
# categories = []
# for x in hot_encoder.categories_:
#     categories.extend(x)
# diamonds_color_clarity_df = pd.DataFrame(diamonds_color_clarity, columns=categories)
# diamondsDF = diamondsDF.join(diamonds_color_clarity_df)
# diamondsDF=diamondsDF.drop(['color','clarity'],axis=1)
# print(diamondsDF)
data = wineDF.loc[:, wineDF.columns != "quality"]
y_data = wineDF[['quality']]
x_train, x_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=100)
y_train, y_test = train_test_split(y_data, test_size=0.2, shuffle=True, random_state=100)
# cut =  diamondsDF.loc[:, diamondsDF.columns == "cut"]
# print(cut)


# In[21]:


# print(x_train)
# print(y_train)


# In[4]:


def plot_data(x, np_arr, title="Figure 1", x_label="Predicted Y", y_label="Test Y", color="blue", label=None):
  plt.plot(x ,np_arr, label=label, color = color)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()


# In[22]:


# DecisionTreeClassifier
print("Running DecisionTreeClassifier:")
decision_learner = DecisionTreeClassifier()
# cross_score = cross_val_score(decision_learner, x_train, y_train,cv =5)
# print(cross_score.mean())
# trained_classifier = decision_learner.fit(x_train, y_train)
# score = decision_learner.score(x_test,y_test)
# score = decision_learner.score(x_train,y_train)
# print(score)


# In[23]:


# Gridsearch
parameters = {'criterion':('gini', 'entropy'), 'max_depth':range(1, 50, 4), 'min_samples_split': range(2,20), 'min_samples_leaf': range(1,10)}
clf = GridSearchCV(decision_learner, parameters)
clf.fit(x_train, y_train)
# print(clf.get_params())
print("Grid search best parameters:")
print(clf.best_params_)


# In[99]:


# learning curve 1
print("Test 1")
decision_learner = DecisionTreeClassifier(criterion='entropy', max_depth = 41)
train_sizes, train_scores, test_scores = learning_curve(decision_learner, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 4 (Decision Tree Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 4 (Decision Tree Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure4.png')
print("Training sizes:")
print(train_sizes)
print("Max cross-validation score:")
# print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print("Max training score at the same point as max cross-validation score:")
print(train_mean[np.argmax(test_mean)])


# In[89]:


# Model complexity analysis
print("Max depth model complexity analysis")
train_scores, test_scores = validation_curve(DecisionTreeClassifier(criterion='entropy'), x_train, y_train, param_name="max_depth", param_range=range(2, 51),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_sizes = list(range(2, 51))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 5 (Decision Tree Validation Curve - Wine Quality)", x_label="Max depth", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 5 (Decision Tree Validation Curve - Wine Quality)", x_label="Max depth", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure5.png')
print('Max depth')
print(np.argmax(test_mean))
print("Max validation score:")
print(test_mean[np.argmax(test_mean)])
# print(np.argmax(train_mean))


# In[102]:


# learning curve 2
print("Test 2")
decision_learner = DecisionTreeClassifier(criterion='entropy', max_depth=18)
train_sizes, train_scores, test_scores = learning_curve(decision_learner, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 6 (Pre-Pruned Decision Tree Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 6 (Pre-Pruned Decision Tree Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure6.png')
print("Training sizes:")
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print("Max training score at the same point as max cross-validation score:")
print(train_mean[np.argmax(test_mean)])


# In[103]:


decision_learner = DecisionTreeClassifier(criterion='entropy', max_depth=18)
trained_classifier = decision_learner.fit(x_train, y_train)
print("final test score")
score = decision_learner.score(x_test,y_test)
print(score)
score = trained_classifier.score(x_train,y_train)
print("final train score")
print(score)


# In[59]:


# AdaBoostClassifier DecisionTreeClassifier
print("Running AdaBoostClassifier:")
boost_learner = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7))
cross_score = cross_val_score(boost_learner, x_train, y_train,cv =5)
print(cross_score.mean())
trained_classifier = boost_learner.fit(x_train, y_train)
# score = learner.score(x_test,y_test)
score = boost_learner.score(x_train,y_train)
print(score)


# In[60]:


# Gridsearch
parameters = {'n_estimators': range(10, 101, 5), 'learning_rate':np.linspace(0.1, 1.5, 15)}
clf = GridSearchCV(boost_learner, parameters)
clf.fit(x_train, y_train)
print(clf.get_params())
print("Grid search best parameters:")
print(clf.best_params_)
# print(clf.score(x_test, y_test))


# In[61]:


# learning curve 1
print('Test 1');
boost_learner = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7), n_estimators=80, learning_rate=0.4)
train_sizes, train_scores, test_scores = learning_curve(boost_learner, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 10 (Boosted Decision Tree Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 10 (Boosted Decision Tree Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure10.png')
print("Training sizes:")
print(train_sizes)
print("Max cross-validation score:")
# print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print("Max training score at the same point as max cross-validation score:")
print(train_mean[np.argmax(test_mean)])


# In[501]:


# print(train_sizes)
# print((train_sizes[np.argmin(train_mean)]))
# print((train_sizes[np.argmax(test_mean)]))
# print(test_mean[np.argmin(train_mean)])
# print(train_mean[np.argmin(train_mean)])


# In[62]:


# Model complexity analysis

print("n_estimators model complexity analysis")
boost_learner = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7),learning_rate=0.4)
train_scores, test_scores = validation_curve(boost_learner, x_train, y_train, param_name="n_estimators", param_range=range(10, 221, 10),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
# train_sizes = list(np.linspace(0.1, 1.5, 15))
train_sizes = list(range(10, 221, 10))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 11 (Boosted Decision Tree Validation Curve - Wine Quality)", x_label="Number of estimators", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 11 (Boosted Decision Tree Validation Curve - Wine Quality)", x_label="Number of estimators", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure11.png')
print(np.argmax(test_mean))
print('n_estimators')
print(np.argmax(test_mean))
print("Max validation score:")
# print(np.argmax(train_mean))


# In[64]:


print(test_mean)


# In[65]:


# learning curve 2
print("Test 2")
boost_learner = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7),n_estimators=180, learning_rate=0.4)
train_sizes, train_scores, test_scores = learning_curve(boost_learner, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 12 (Boosted Decision Tree Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 12 (Boosted Decision Tree Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure12.png')
print("Training sizes:")
print(train_sizes)
print("Max cross-validation score:")
# print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print("Max training score at the same point as max cross-validation score:")
print(train_mean[np.argmax(test_mean)])


# In[66]:


boost_learner = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7),n_estimators=180, learning_rate=0.4 )
trained_boost_classifier = boost_learner.fit(x_train, y_train)
score = trained_boost_classifier.score(x_test,y_test)
print("final test score")
print(score)
score = trained_boost_classifier.score(x_train,y_train)
print("final train score")
print(score)


# In[5]:


scaler = StandardScaler()  # doctest: +SKIP
# Don't cheat - fit only on training data
scaler.fit(x_train)  # doctest: +SKIP
x_train_scaled = scaler.transform(x_train)  # doctest: +SKIP
# apply same transformation to test data
x_test_scaled = scaler.transform(x_test)  # doctest: +SKIP


# In[68]:


# MLPClassifier
print("Running MLPClassifier:")
mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='logistic')
cross_score = cross_val_score(mlp_learner, x_train_scaled, y_train,cv =5)
print(cross_score.mean())
trained_classifier = mlp_learner.fit(x_train_scaled, y_train)
# score = learner.score(x_test,y_test)
score = mlp_learner.score(x_train ,y_train)
score_scaled = mlp_learner.score(x_train_scaled ,y_train)
print(score)
print(score_scaled)


# In[69]:


# Gridsearch
parameters = {'hidden_layer_sizes': [(13,),(100,),(10,10),(10,10,10)], 'activation':['identity', 'logistic', 'tanh', 'relu'], 'learning_rate_init': np.linspace(0.001, 0.1, 10), 'learning_rate':['constant', 'invscaling', 'adaptive'],'solver':['sgd','adam']}
clf = GridSearchCV(mlp_learner, parameters)
clf.fit(x_train_scaled, y_train)
# print(clf.get_params())
print(clf.best_params_)
print(clf.score(x_test_scaled, y_test))


# In[70]:


print(clf.best_params_)
print(clf.score(x_test_scaled, y_test))


# In[143]:


# learning curve 1
mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 16 (Neural Network Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 16 (Neural Network Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure16.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[141]:


# Model complexity analysis
mlp_learner = MLPClassifier(hidden_layer_sizes=(150,150),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_scores, test_scores = validation_curve(mlp_learner, x_train_scaled, y_train, param_name="hidden_layer_sizes", param_range=[(x,y) for x,y in zip(range(5, 221, 10),range(5, 221, 10))],
#     scoring="accuracy", n_jobs=-1, cv=5)
# train_scores, test_scores = validation_curve(mlp_learner, x_train_scaled, y_train, param_name="alpha", param_range=np.linspace(0.0001, 0.3, 10),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_sizes = list(range(5, 221, 10))
# train_sizes = [(3,3),(4,4)]
# train_sizes = list(np.linspace(0.0001, 0.3, 10))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 17 (Neural Network Validation Curve - Wine Quality)", x_label="Number of hidden layer nodes", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 17 (Neural Network Validation Curve - Wine Quality)", x_label="Number of hidden layer nodes", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure17.png')
print(np.argmax(test_mean))
print(test_mean[np.argmax(test_mean)])
print(np.argmax(train_mean))


# In[106]:


# print(np.linspace(0.001, 1.2, 60))
# print(list(range(3, 21, 4)))
[(x,y) for x,y in zip(range(5, 221, 10),range(5, 221, 10))]
# [(x,) for x in range(5, 221, 10)]


# In[150]:


# learning curve 2
mlp_learner = mlp_learner = MLPClassifier(hidden_layer_sizes=(180, 180),activation='logistic',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07, alpha=0.2, early_stopping=False)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 18 (Neural Network Learning Curve 2 - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 18 (Neural Network Learning Curve 2- Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure18.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[149]:


mlp_learner = mlp_learner = MLPClassifier(hidden_layer_sizes=(180, 180),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07, alpha=0.2, early_stopping=False)
trained_mlp_classifier = mlp_learner.fit(x_train_scaled, y_train)
score = trained_mlp_classifier.score(x_test_scaled,y_test)
print(score)


# In[127]:


# KNeighborsClassifier
print("Running KNeighborsClassifier:")
knn_learner = KNeighborsClassifier()
cross_score = cross_val_score(knn_learner, x_train, y_train, cv=5)
print(cross_score.mean())
trained_classifier = knn_learner.fit(x_train_scaled, y_train)
# score = learner.score(x_test_scaled,y_test)
score = knn_learner.score(x_train_scaled ,y_train)
print(score)


# In[128]:


# Gridsearch
parameters = {'n_neighbors': range(5, 126, 20), 'weights':['uniform','distance']}
clf = GridSearchCV(knn_learner, parameters)
clf.fit(x_train_scaled, y_train)
# print(clf.get_params())
print(clf.best_params_)
print(clf.score(x_test_scaled, y_test))


# In[549]:


print(list(range(1, 51, 4)))


# In[144]:


# learning curve 1
knn_learner = KNeighborsClassifier(n_neighbors=31, weights='uniform')
train_sizes, train_scores, test_scores = learning_curve(knn_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 22 (kNN Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 22 (kNN Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure22.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[ ]:





# In[133]:


# Model complexity analysis
knn_learner = KNeighborsClassifier(weights='uniform')
train_scores, test_scores = validation_curve(knn_learner, x_train_scaled, y_train, param_name="n_neighbors", param_range=range(5, 71, 5),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
# train_sizes = list(np.linspace(0.1, 1.5, 15))
train_sizes = list(range(5, 71, 5))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 23 (kNN Validation Curve - Wine Quality)", x_label="Number of neighbors (k)", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 23 (kNN Validation Curve - Wine Quality)", x_label="Number of neighbors (k)", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure23.png')
print(np.argmax(test_mean))
print(test_mean[np.argmax(test_mean)])
print(np.argmax(train_mean))


# In[148]:


# learning curve 2
knn_learner = KNeighborsClassifier(n_neighbors=31, weights='distance')
train_sizes, train_scores, test_scores = learning_curve(knn_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 24 (kNN Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 24 (kNN Learning Curve - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure24.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[146]:


trained_knn_learner = knn_learner.fit(x_train_scaled, y_train)
score = knn_learner.score(x_test_scaled,y_test)
print(score)


# In[7]:


# SVC
print("Running SVC:")
svc_learner = SVC()
cross_score = cross_val_score(svc_learner, x_train_scaled, y_train, cv=5)
print(cross_score.mean())
trained_classifier = svc_learner.fit(x_train_scaled, y_train)
# score = learner.score(x_test,y_test)
score = svc_learner.score(x_train_scaled ,y_train)
print(score)


# In[10]:


# Gridsearch
parameters = {'C': np.linspace(0.1, 30, 30), 'kernel':['rbf','linear','poly','rbf','sigmoid']}
clf = GridSearchCV(svc_learner, parameters)
clf.fit(x_train_scaled, y_train)
# print(clf.get_params())
print(clf.best_params_)
print(clf.score(x_test_scaled, y_test))


# In[566]:


print(clf.best_params_)
print(clf.score(x_test_scaled, y_test))


# In[11]:


# learning curve 1
svc_learner = SVC(C=25)
train_sizes, train_scores, test_scores = learning_curve(svc_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 29 (SVM Learning Curve [kernel=rbf] - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 29 (SVM Learning Curve [kernel=rbf] - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure29.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[12]:


# Model complexity analysis
svc_learner = SVC(kernel='linear')
train_scores, test_scores = validation_curve(svc_learner, x_train_scaled, y_train, param_name="C", param_range=np.linspace(0.5, 3.5, 6),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_sizes = list(np.linspace(0.5, 3.5, 6))
# train_sizes = list(range(10, 121, 5))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 30 (SVM Validation Curve [kernel=linear] - Wine Quality)", x_label="C", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 30 (SVM Validation Curve [kernel=linear] - Wine Quality)", x_label="C", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure30.png')
print(np.argmax(test_mean))
print(test_mean[np.argmax(test_mean)])
print(np.argmax(train_mean))


# In[13]:


# Model complexity analysis
svc_learner = SVC(kernel='rbf')
train_scores, test_scores = validation_curve(svc_learner, x_train_scaled, y_train, param_name="C", param_range=np.linspace(1, 50, 10),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_sizes = list(np.linspace(1, 50, 10))
# train_sizes = list(range(10, 121, 5))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 31 (SVM Validation Curve [kernel=rbf] - Wine Quality)", x_label="C", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 31 (SVM Validation Curve [kernel=rbf] - Wine Quality)", x_label="C", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure31.png')
print(np.argmax(test_mean))
print(test_mean[np.argmax(test_mean)])
print(np.argmax(train_mean))


# In[575]:


print(train_sizes)


# In[17]:


# learning curve 2
svc_learner = SVC(C=20)
train_sizes, train_scores, test_scores = learning_curve(svc_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 32 (SVM Learning Curve [kernel=rbf] - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 32 (SVM Learning Curve [kernel=rbf] - Wine Quality)", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure32.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[19]:


trained_svc_learner = svc_learner.fit(x_train_scaled, y_train)
score = svc_learner.score(x_test_scaled,y_test)
print(score)


# In[ ]:




