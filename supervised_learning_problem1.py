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


# In[2]:


# DIAMONDS data analysis


# In[3]:


# Load dataset

diamondsDF = pd.read_csv('diamonds.csv')


# In[358]:


# print(diamondsDF)


# In[4]:


# pre-process dataset
ord_encoder = OrdinalEncoder(categories=[['Fair','Good','Very Good','Premium','Ideal']])
diamondsDF["cut"] = ord_encoder.fit_transform(diamondsDF[['cut']])
# print(transformed_cut)
#  = transformed_cut
# print(diamondsDF)
hot_encoder = OneHotEncoder(handle_unknown='ignore')
diamonds_color_clarity = hot_encoder.fit_transform(diamondsDF[['color','clarity']]).toarray()
categories = []
for x in hot_encoder.categories_:
    categories.extend(x)
diamonds_color_clarity_df = pd.DataFrame(diamonds_color_clarity, columns=categories)
diamondsDF = diamondsDF.join(diamonds_color_clarity_df)
diamondsDF=diamondsDF.drop(['color','clarity'],axis=1)
# print(diamondsDF)
data = diamondsDF.loc[:, diamondsDF.columns != "cut"]
y_data = diamondsDF[['cut']]
x_train, x_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=100)
y_train, y_test = train_test_split(y_data, test_size=0.2, shuffle=True, random_state=100)
# cut =  diamondsDF.loc[:, diamondsDF.columns == "cut"]
# print(cut)


# In[5]:


def plot_data(x, np_arr, title="Figure 1", x_label="Predicted Y", y_label="Test Y", color="blue", label=None):
  plt.plot(x ,np_arr, label=label, color = color)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()


# In[412]:


# DecisionTreeClassifier
print("Running DecisionTreeClassifier:")
decision_learner = DecisionTreeClassifier()
# cross_score = cross_val_score(decision_learner, x_train, y_train,cv =5)
# print(cross_score.mean())
# trained_classifier = decision_learner.fit(x_train, y_train)
# score = decision_learner.score(x_test,y_test)
# score = decision_learner.score(x_train,y_train)
# print(score)


# In[414]:


# Gridsearch
parameters = {'criterion':('gini', 'entropy'), 'max_depth':range(1, 50, 4), 'min_samples_split': range(2,20), 'min_samples_leaf': range(1,10)}
clf = GridSearchCV(decision_learner, parameters)
clf.fit(x_train, y_train)
# print(clf.get_params())
print("Grid search best parameters:")
print(clf.best_params_)


# In[416]:


# learning curve 1
print("Test 1")
decision_learner = DecisionTreeClassifier(criterion='entropy', max_depth = 13)
train_sizes, train_scores, test_scores = learning_curve(decision_learner, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 1", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 1", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure1.png')
print("Training sizes:")
print(train_sizes)
print("Max cross-validation score:")
# print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print("Max training score at the same point as max cross-validation score:")
print(train_mean[np.argmax(test_mean)])


# In[439]:


# Model complexity analysis
print("Max depth model complexity analysis")
train_scores, test_scores = validation_curve(DecisionTreeClassifier(criterion='entropy'), x_train, y_train, param_name="max_depth", param_range=range(2, 21),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_sizes = list(range(2, 21))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 2", x_label="Max depth", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 2", x_label="Max depth", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure2.png')
print('Max depth')
print(np.argmax(test_mean))
print("Max validation score:")
print(test_mean[np.argmax(test_mean)])
# print(np.argmax(train_mean))


# In[441]:


# learning curve 2
print("Test 2")
decision_learner = DecisionTreeClassifier(criterion='entropy', max_depth=9)
train_sizes, train_scores, test_scores = learning_curve(decision_learner, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 3", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 3", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure3.png')
print("Training sizes:")
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print("Max training score at the same point as max cross-validation score:")
print(train_mean[np.argmax(test_mean)])


# In[543]:


decision_learner = DecisionTreeClassifier(criterion='entropy', max_depth=9)
trained_classifier = decision_learner.fit(x_train, y_train)
print("final test score")
score = decision_learner.score(x_test,y_test)
print(score)
score = trained_classifier.score(x_train,y_train)
print("final train score")
print(score)


# In[448]:


# AdaBoostClassifier DecisionTreeClassifier
print("Running AdaBoostClassifier:")
boost_learner = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=29))
cross_score = cross_val_score(boost_learner, x_train, y_train,cv =5)
# print(cross_score.mean())
trained_classifier = boost_learner.fit(x_train, y_train)
# score = learner.score(x_test,y_test)
score = boost_learner.score(x_train,y_train)
print(score)


# In[446]:


# Gridsearch
parameters = {'n_estimators': range(10, 71, 5), 'learning_rate':np.linspace(0.1, 1.5, 15)}
clf = GridSearchCV(boost_learner, parameters)
clf.fit(x_train, y_train)
# print(clf.get_params())
print(clf.best_params_)
print(clf.score(x_test, y_test))


# In[497]:


# learning curve 1
boost_learner = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9))
train_sizes, train_scores, test_scores = learning_curve(boost_learner, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 7", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 7", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure7.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[501]:


print(train_sizes)
print((train_sizes[np.argmin(train_mean)]))
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmin(train_mean)])
print(train_mean[np.argmin(train_mean)])


# In[579]:


# Model complexity analysis
boost_learner = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9))
train_scores, test_scores = validation_curve(boost_learner, x_train, y_train, param_name="n_estimators", param_range=range(10, 121, 5),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
# train_sizes = list(np.linspace(0.1, 1.5, 15))
train_sizes = list(range(10, 121, 5))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 8", x_label="Number of estimators", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 8", x_label="Number of estimators", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure8.png')
print(np.argmax(test_mean))
print(test_mean[np.argmax(test_mean)])
print(np.argmax(train_mean))


# In[580]:


# learning curve 2
boost_learner = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9),n_estimators=120, learning_rate=1 )
train_sizes, train_scores, test_scores = learning_curve(boost_learner, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 9", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 9", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure9.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[581]:


boost_learner = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9),n_estimators=120, learning_rate=1 )
trained_boost_classifier = boost_learner.fit(x_train, y_train)
score = trained_boost_classifier.score(x_test,y_test)
print(score)
score = trained_boost_classifier.score(x_train,y_train)
print(score)


# In[6]:


scaler = StandardScaler()  # doctest: +SKIP
# Don't cheat - fit only on training data
scaler.fit(x_train)  # doctest: +SKIP
x_train_scaled = scaler.transform(x_train)  # doctest: +SKIP
# apply same transformation to test data
x_test_scaled = scaler.transform(x_test)  # doctest: +SKIP


# In[490]:


# MLPClassifier
mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='logistic')
cross_score = cross_val_score(mlp_learner, x_train_scaled, y_train,cv =5)
print(cross_score.mean())
trained_classifier = mlp_learner.fit(x_train_scaled, y_train)
# score = learner.score(x_test,y_test)
score = mlp_learner.score(x_train ,y_train)
score_scaled = mlp_learner.score(x_train_scaled ,y_train)
print(score)
print(score_scaled)


# In[502]:


# Gridsearch
parameters = {'hidden_layer_sizes': [(13,),(100,),(10,10)], 'activation':['identity', 'logistic', 'tanh', 'relu'], 'learning_rate_init': np.linspace(0.001, 0.1, 10), 'learning_rate':['constant', 'invscaling', 'adaptive'],'solver':['sgd','adam']}
clf = GridSearchCV(mlp_learner, parameters)
clf.fit(x_train_scaled, y_train)
# print(clf.get_params())
print(clf.best_params_)
print(clf.score(x_test_scaled, y_test))


# In[504]:


print(clf.best_params_)
print(clf.score(x_test_scaled, y_test))


# In[506]:


# learning curve 1
mlp_learner = MLPClassifier(hidden_layer_sizes=(10,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 13", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 13", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure13.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[536]:


# Model complexity analysis
mlp_learner = MLPClassifier(hidden_layer_sizes=(14,14),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
# train_scores, test_scores = validation_curve(mlp_learner, x_train_scaled, y_train, param_name="hidden_layer_sizes", param_range=[(x,) for x in range(3, 21, 4)],
#     scoring="accuracy", n_jobs=-1, cv=5)
train_scores, test_scores = validation_curve(mlp_learner, x_train_scaled, y_train, param_name="learning_rate_init", param_range=np.linspace(0.001, 0.1, 15),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
# train_sizes = list(range(3, 21, 4))
train_sizes = list(np.linspace(0.001, 0.15, 15))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 14", x_label="Number of hidden layer nodes", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 14", x_label="Number of hidden layer nodes", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure14.png')
print(np.argmax(test_mean))
print(test_mean[np.argmax(test_mean)])
print(np.argmax(train_mean))


# In[537]:


print(np.linspace(0.001, 0.1, 15))
print(list(range(3, 21, 4)))
# [(x,y) for x,y in zip(range(7, 99),range(0, 20))]
[(x,) for x in range(3, 121, 8)]


# In[534]:


# learning curve 2
mlp_learner = mlp_learner = MLPClassifier(hidden_layer_sizes=(14, 14),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 15", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 15", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure15.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[541]:


trained_mlp_classifier = mlp_learner.fit(x_train_scaled, y_train)
score = trained_mlp_classifier.score(x_test_scaled,y_test)
print(score)


# In[545]:


# KNeighborsClassifier
knn_learner = KNeighborsClassifier()
cross_score = cross_val_score(knn_learner, x_train, y_train, cv=5)
print(cross_score.mean())
trained_classifier = knn_learner.fit(x_train_scaled, y_train)
# score = learner.score(x_test_scaled,y_test)
score = knn_learner.score(x_train_scaled ,y_train)
print(score)


# In[556]:


# Gridsearch
parameters = {'n_neighbors': range(5, 126, 20), 'weights':['uniform','distance']}
clf = GridSearchCV(knn_learner, parameters)
clf.fit(x_train_scaled, y_train)
# print(clf.get_params())
print(clf.best_params_)
print(clf.score(x_test_scaled, y_test))


# In[549]:


print(list(range(1, 51, 4)))


# In[560]:


# learning curve 1
knn_learner = KNeighborsClassifier(n_neighbors=25, weights='uniform')
train_sizes, train_scores, test_scores = learning_curve(knn_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 19", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 19", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure19.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[ ]:





# In[558]:


# Model complexity analysis
knn_learner = KNeighborsClassifier(weights='uniform')
train_scores, test_scores = validation_curve(knn_learner, x_train_scaled, y_train, param_name="n_neighbors", param_range=range(5, 51, 5),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
# train_sizes = list(np.linspace(0.1, 1.5, 15))
train_sizes = list(range(5, 51, 5))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 20", x_label="Number of neighbors (k)", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 20", x_label="Number of neighbors (k)", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure20.png')
print(np.argmax(test_mean))
print(test_mean[np.argmax(test_mean)])
print(np.argmax(train_mean))


# In[559]:


# learning curve 2
knn_learner = KNeighborsClassifier(n_neighbors=10, weights='uniform')
train_sizes, train_scores, test_scores = learning_curve(knn_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 21", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 21", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure21.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[561]:


trained_knn_learner = knn_learner.fit(x_train_scaled, y_train)
score = knn_learner.score(x_test_scaled,y_test)
print(score)


# In[563]:


# SVC
svc_learner = SVC()
cross_score = cross_val_score(svc_learner, x_train_scaled, y_train, cv=5)
print(cross_score.mean())
trained_classifier = svc_learner.fit(x_train_scaled, y_train)
# score = learner.score(x_test,y_test)
score = svc_learner.score(x_train_scaled ,y_train)
print(score)


# In[565]:


# Gridsearch
parameters = {'C': np.linspace(0.1, 1.5, 15), 'kernel':['rbf','linear','poly','rbf','sigmoid']}
clf = GridSearchCV(svc_learner, parameters)
clf.fit(x_train_scaled, y_train)
# print(clf.get_params())
print(clf.best_params_)
print(clf.score(x_test_scaled, y_test))


# In[566]:


print(clf.best_params_)
print(clf.score(x_test_scaled, y_test))


# In[567]:


# learning curve 1
svc_learner = SVC(C=1.5)
train_sizes, train_scores, test_scores = learning_curve(svc_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 25", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 25", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure25.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[ ]:


# Model complexity analysis
svc_learner = SVC(kernel='linear')
train_scores, test_scores = validation_curve(svc_learner, x_train_scaled, y_train, param_name="C", param_range=np.linspace(0.5, 3.5, 6),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_sizes = list(np.linspace(0.5, 3.5, 6))
# train_sizes = list(range(10, 121, 5))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 26", x_label="C", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 26", x_label="C", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure26.png')
print(np.argmax(test_mean))
print(test_mean[np.argmax(test_mean)])
print(np.argmax(train_mean))


# In[ ]:


# Model complexity analysis
svc_learner = SVC(kernel='rbf')
train_scores, test_scores = validation_curve(svc_learner, x_train_scaled, y_train, param_name="C", param_range=np.linspace(0.5, 3.5, 6),
    scoring="accuracy", n_jobs=-1, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_sizes = list(np.linspace(0.5, 3.5, 6))
# train_sizes = list(range(10, 121, 5))
plt.close()
plot_data(train_sizes, test_mean, title="Figure 27", x_label="C", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 27", x_label="C", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure27.png')
print(np.argmax(test_mean))
print(test_mean[np.argmax(test_mean)])
print(np.argmax(train_mean))


# In[7]:


# learning curve 2
svc_learner = SVC(kernel='linear')
train_sizes, train_scores, test_scores = learning_curve(svc_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.close()
plot_data(train_sizes, test_mean, title="Figure 28", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Cross-validation score')
plot_data(train_sizes, train_mean, title="Figure 28", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training score')
plt.savefig('figure28.png')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])


# In[8]:


# svc_learner = SVC(kernel='linear')
trained_svc_learner = svc_learner.fit(x_train_scaled, y_train)
score = knn_learner.score(x_test_scaled,y_test)
print(score)


# In[582]:


print(np.linspace(0.5, 3.5, 6))


# In[ ]:




