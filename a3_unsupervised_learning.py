"""
ML (CS7641) - Unsupervised Learning and Dimensionality Reduction
Student Name: Gevorg Ter-Petrosyan
GT User ID: gterpetrosyan3
"""


#!/usr/bin/env python
# coding: utf-8

# In[1087]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import homogeneity_score,normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, v_measure_score, homogeneity_completeness_v_measure, silhouette_score 
from scipy.stats import kurtosis,entropy
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
import time
from sklearn.neural_network import MLPClassifier
import warnings
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')


# In[1088]:


def plot_data(x, np_arr, title="Figure 1", x_label="Predicted Y", y_label="Test Y", color="blue", label=None, linestyle = 'solid', handlelength=3):
  plt.plot(x ,np_arr, label=label, color = color, linestyle = linestyle )
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(handlelength=handlelength)


# In[1089]:


random_state = 101

def km_em(x_train_scaled, dataset_name="", true_vals = y_train, reg_covar = 1e-01):
    distortions = []
    sil = []
    n = 22
    # v_measure = []
    homogeneity = []
    completeness = []
    mutual_info = []
    adj_rand_score = []
    sil = []
    kmeans_times = []
    homogeneity_em = []
    completeness_em = []
    mutual_info_em = []
    adj_rand_score_em = []
    sil_em = []
    em_times = []
    em_likelihood = []
    for i in range(2,n+1):
#         print(i)
        start_time = time.time()
        kmeans = KMeans(n_clusters=i, random_state=random_state)
        kmeans.fit(x_train_scaled)
        distortions.append(kmeans.inertia_)
        y_pred = kmeans.predict(x_train_scaled)
        kmeans_times.append(time.time()-start_time)
        homogeneity.append(homogeneity_score(true_vals, y_pred.tolist()))
        completeness.append(completeness_score(true_vals, y_pred.tolist()))
        mutual_info.append(adjusted_mutual_info_score(true_vals, y_pred.tolist()))
        adj_rand_score.append(adjusted_rand_score(true_vals, y_pred.tolist()))
        sil.append(silhouette_score(x_train_scaled, kmeans.labels_, metric='euclidean'))
        start_time = time.time()
        gm = BayesianGaussianMixture(n_components = i, random_state=random_state, reg_covar=reg_covar)
        y_pred = gm.fit_predict(x_train_scaled)
        em_times.append(time.time()-start_time)
        homogeneity_em.append(homogeneity_score(true_vals, y_pred.tolist()))
        completeness_em.append(completeness_score(true_vals, y_pred.tolist()))
        mutual_info_em.append(adjusted_mutual_info_score(true_vals, y_pred.tolist()))
        adj_rand_score_em.append(adjusted_rand_score(true_vals, y_pred.tolist()))
        if len(set(y_pred))>1:
            sil_em.append(silhouette_score(x_train_scaled, y_pred, metric='euclidean'))
        else:
            sil_em.append(1)
        em_likelihood.append(gm.score(x_train_scaled))
    # plot
    plt.plot(range(2, n+1), distortions, marker='o')
    plt.title("K-means Elbow ("+(str(dataset_name))+")")
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.savefig((str(dataset_name))+' km elbow.png')
    plt.show()

    plt.plot(range(2, n+1), sil, marker='o')
    plt.title('K-means Silhouette Scores ('+(str(dataset_name))+')')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig((str(dataset_name))+' km silho.png')
    plt.show()

    plt.plot(range(2, n+1), em_likelihood, marker='o')
    plt.title('EM likelihood ('+(str(dataset_name))+')')
    plt.xlabel('Number of clusters')
    plt.ylabel('Likelihood')
    plt.savefig((str(dataset_name))+' em likelihood.png')
    plt.show()
    
    plt.plot(range(2, n+1), sil_em, marker='o')
    plt.title('EM Silhouette Scores ('+(str(dataset_name))+')')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig((str(dataset_name))+' em silho.png')
    plt.show()
    
    plt.close()
    plot_data(list(range(1, n)), homogeneity, title="Performance Evaluation k-means ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Score", color="blue", label='Homogeneity')
    plot_data(list(range(1, n)), completeness, title="Performance Evaluation k-means ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Score", color="orange", label='Completeness')
    plot_data(list(range(1, n)), mutual_info, title="Performance Evaluation k-means ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Score", color="red", label='Adgusted Mutual Info')
    plot_data(list(range(1, n)), adj_rand_score, title="Performance Evaluation k-means ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Score", color="green", label='Adjusted random index')
    # plot_data(list(range(1, n)), v_measure, title="Performance Evaluation k-means", x_label="Number of Clusters", y_label="Score", color="brown", label='V-measure')
    plt.savefig((str(dataset_name))+' km perfo.png')
    plt.show()

    plt.close()
    plot_data(list(range(1, n)), homogeneity_em, title="Performance Evaluation EM ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Score", color="blue", label='Homogeneity')
    plot_data(list(range(1, n)), completeness_em, title="Performance Evaluation EM ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Score", color="orange", label='Completeness')
    plot_data(list(range(1, n)), mutual_info_em, title="Performance Evaluation EM ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Score", color="red", label='Adgusted Mutual Info')
    plot_data(list(range(1, n)), adj_rand_score_em, title="Performance Evaluation EM ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Score", color="green", label='Adjusted random index')
    # plot_data(list(range(1, n)), v_measure, title="Performance Evaluation EM", x_label="Number of Clusters", y_label="Score", color="brown", label='V-measure')
    plt.savefig((str(dataset_name))+' em perfo.png')
    plt.show()

    plt.close()
    plot_data(list(range(1, n)), kmeans_times, title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="red", label='k-means')
    plot_data(list(range(1, n)), em_times, title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="blue", label='EM')
    plt.savefig((str(dataset_name))+' km-em time.png')
    plt.show()
    print('kmeans_times')
    print(kmeans_times)
    print('em_times')
    print(em_times)
    
    return {'sil': sil, 'kmeans_times':kmeans_times, 'em_times':em_times, 'homogeneity':homogeneity, 'completeness':completeness, 'mutual_info':mutual_info, 'adj_rand_score':adj_rand_score, 'homogeneity_em':homogeneity_em, 'completeness_em':completeness_em, 'mutual_info_em':mutual_info_em, 'adj_rand_score_em':adj_rand_score_em}


# In[1090]:


def plot_clusters_3d(x_train_scaled, y_pred, x, y, z, dataset_name="", classifier = "K-means"):        
#     fig = plt.figure()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    x_vals =x_train_scaled[:, x]
    y_vals =x_train_scaled[:, y]
    z_vals =x_train_scaled[:, z]
    scatter = ax.scatter(x_vals, y_vals, z_vals, c=y_pred, marker='o')
    ax.set_xlabel('Feature '+str(x))
    ax.set_ylabel('Feature '+str(y))
    ax.set_zlabel('Feature '+str(z))
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.title(classifier + " Clustering ("+(str(dataset_name))+")")
    plt.savefig((str(dataset_name))+' '+classifier+' Clustering.png')
    plt.show()


# In[1091]:


def plot_clusters(x_train_scaled, y_pred, x, y, dataset_name="", classifier = "K-means"):        
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.scatter(x_train_scaled[:, x], x_train_scaled[:, y], c=y_pred)
    plt.title(classifier + " Clustering ("+(str(dataset_name))+")")
    plt.savefig((str(dataset_name))+' '+classifier+' Clustering.png')
    plt.show()


# In[1092]:


def plot_km_em_clusters_2d(x_train_scaled, x=0, y=1, z = 0, dataset_name = "", km_clusters = 12, em_clusters=12):
    kmeans = KMeans(n_clusters=km_clusters, random_state=random_state)
    y_pred = kmeans.fit_predict(x_train_scaled)
    plot_clusters(x_train_scaled, y_pred, x, y, dataset_name=dataset_name, classifier = "K-means")
    gm = BayesianGaussianMixture(n_components = em_clusters, random_state=random_state, reg_covar=1e-01)
    y_pred = gm.fit_predict(x_train_scaled)
    plot_clusters(x_train_scaled, y_pred, x, y,  dataset_name=dataset_name, classifier = "EM")


# In[1093]:


def plot_km_em_clusters(x_train_scaled, x=0, y=1, z = 0, dataset_name = "",  km_clusters = 12, em_clusters=12):
    kmeans = KMeans(n_clusters=km_clusters, random_state=random_state)
    y_pred = kmeans.fit_predict(x_train_scaled)
    plot_clusters_3d(x_train_scaled, y_pred, x, y, z, dataset_name=dataset_name, classifier = "K-means")
    gm = BayesianGaussianMixture(n_components = em_clusters, random_state=random_state, reg_covar=1e-01)
    y_pred = gm.fit_predict(x_train_scaled)
    plot_clusters_3d(x_train_scaled, y_pred, x, y, z, dataset_name=dataset_name, classifier = "EM")


# In[1094]:


# PCA
def pca_dim_red(x_train_scaled, dataset_name, features_num = 15 ):
    pca = PCA(random_state=random_state)
    pca_result = pca.fit_transform(x_train_scaled)
    # print(pca.explained_variance_ratio_)
    vals = pca.explained_variance_ratio_
    plt.bar(list(range(len(vals))),vals)
    plt.title("PCA Principal Component Variance Ratios ("+str(dataset_name)+")")
    plt.ylabel("Variance Ratio")
    plt.xlabel("Component")
    tot = []
    for c in vals:
        tot.append(c)
        if sum(tot)>=0.9:
            break
    plt.savefig((str(dataset_name))+' pca analysis.png')
    plt.show()
    print("num of features >90% var")
    print(len(tot))
    print(1-sum(tot))
    pca = PCA(n_components=features_num, random_state=random_state)
    pca_result = pca.fit_transform(x_train_scaled)
    print('pca_result.shape')
    print(pca_result.shape)
    x_projected_pca = pca.inverse_transform(pca_result)
    print('x_projected.shape')
    print(x_projected_pca.shape)
    print('x_train_scaled.shape')
    print(x_train_scaled.shape)
    loss = ((x_train_scaled - x_projected_pca) ** 2).mean()
    print("loss (x_train_scaled - x_projected_pca) ** 2).mean()")
    print(loss)
    return pca_result,x_projected_pca


# In[1095]:


# ICA
def ica_dim_red(x_train_scaled, dataset_name, features_num = 12):
    ica = FastICA(random_state=random_state)
    temp = ica.fit_transform(x_train_scaled)
    order = [-abs(kurtosis(temp[:,i])) for i in range(temp.shape[1])]
    temp = temp[:,np.array(order).argsort()]
    ica_res =  pd.Series([abs(kurtosis(temp[:,i])) for i in range(temp.shape[1])]);
    
    l = plt.bar(list(range(len(ica_res))),ica_res, log = True)
    plt.title("ICA Feature Kurtosis ("+str(dataset_name)+")")
    plt.ylabel("Kurtosis")
    plt.xlabel("Features (ordered by kurtosis)")
    plt.savefig((str(dataset_name))+' ica analysis.png')
    plt.show()
    # print(temp)
    
    print("List of of features with kurtosis <= 3")    
    print(np.where(np.log10(ica_res) <= 0.5)[0])
    
    
    ica = FastICA(n_components=features_num, random_state=random_state)
    ica_result = ica.fit_transform(x_train_scaled)
    print(ica_result.shape)
    x_projected_ica = ica.inverse_transform(ica_result)
    print(x_projected.shape)
    print(x_train_scaled.shape)
    loss = ((x_train_scaled - x_projected_ica) ** 2).mean()
    print(loss)
    return ica_result,x_projected_ica


# In[1096]:


# Randomized Projections
def rp_dim_red(x_train_scaled, dataset_name, features_num = 19):
    print(x_train_scaled.shape[1])
    rp_feature_losses = []
    rp_feature_stds = []
    z=0
    for k in range(1, x_train_scaled.shape[1]+1):
        losses = []
        for m in range(5):
            rp = GaussianRandomProjection(k)
            rp_result = rp.fit_transform(x_train_scaled)
            # inverse_transform
            inv = np.linalg.pinv(rp.components_.T)
            x_projected_rp = rp_result.dot(inv)
            loss = ((x_train_scaled - x_projected_rp) ** 2).mean()
            losses.append(loss)
        rp_feature_stds.append(np.std(losses))   
        rp_feature_losses.append(np.mean(losses))

    np_feature_losses_percent = np.multiply(100, rp_feature_losses/np.sum(rp_feature_losses))
    print("std")
    print(rp_feature_stds)
    print('loss')
    print(rp_feature_losses)
    print('sum')
    print(np.sum(rp_feature_losses))
    print('%')
    print(np_feature_losses_percent)
    print('num of clustrs < 10% loss')
    for i in range(len(np_feature_losses_percent)):
        z=z+np_feature_losses_percent[i]
        if z>90:
            print(i+1)
            break    
    plt.bar(list(range(1,len(np_feature_losses_percent)+1)), np_feature_losses_percent)
    plt.title("Random Projection Losses % ("+str(dataset_name)+")")
    plt.ylabel("Mean Squared Error (% of Total)")
    plt.xlabel("Features")
    plt.savefig((str(dataset_name))+' rp analysis % loss.png')
    plt.show()
    
    plt.bar(list(range(1,len(rp_feature_losses)+1)), rp_feature_losses)
    plt.title("Random Projection Losses ("+str(dataset_name)+")")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Features")
    # plt.subplots_adjust(bottom=.15, left=.15)
    plt.savefig((str(dataset_name))+' rp analysis.png')
    plt.show()
    
    plt.bar(list(range(1,len(rp_feature_stds)+1)), rp_feature_stds)
    plt.title("Random Projection STDs ("+str(dataset_name)+")")
    plt.ylabel("STD")
    plt.xlabel("Features")
    plt.savefig((str(dataset_name))+' rp std analysis.png')
    plt.show()
    
    rp = GaussianRandomProjection(features_num,random_state=random_state)
    rp_result = rp.fit_transform(x_train_scaled)
    inv = np.linalg.pinv(rp.components_.T)
    x_projected_rp = rp_result.dot(inv)
    return rp_result, x_projected_rp


# In[1097]:


#FA
def fa_dim_red(x_train_scaled, dataset_name, features_num = 2):
    z=0
    losses = []
    for k in range(1, x_train_scaled.shape[1]+1):
        fa = FeatureAgglomeration(n_clusters=k)
        fa_result = fa.fit_transform(x_train_scaled)
        x_projected_fa = fa.inverse_transform(fa_result)
        loss = ((x_train_scaled - x_projected_fa) ** 2).mean()
        losses.append(loss)
            
    np_feature_losses_percent = np.multiply(100, losses/np.sum(losses))
    print('num of clustrs < 10% loss')
    for i in range(len(np_feature_losses_percent)):
        z=z+np_feature_losses_percent[i]
        if z>90:
            print(i+1)
            break
    print(np_feature_losses_percent)
    plt.bar(list(range(1,len(np_feature_losses_percent)+1)),np_feature_losses_percent)
    plt.title("FeatureAgglomeration Projection Losses % ("+str(dataset_name)+")")
    plt.ylabel("Mean Squared Error (% of Total)")
    plt.xlabel("Features")
    plt.savefig((str(dataset_name))+' fa analysis.png')
    plt.show()

    fa = FeatureAgglomeration(n_clusters=features_num)
    fa_result = fa.fit_transform(x_train_scaled, y_train)
    print(fa_result.shape)
    x_projected_fa = fa.inverse_transform(fa_result)
    print(x_projected_ica.shape)
    print(x_train_scaled.shape)
    loss = ((x_train_scaled - x_projected_fa) ** 2).mean()
    print('loss')
    print(loss)
    return fa_result,x_projected_fa


# In[1098]:


# Dataset 1 Diamonds

diamondsDF = pd.read_csv('diamonds.csv', index_col=[0])
print(diamondsDF)
# pre-process dataset
ord_encoder = OrdinalEncoder(categories=[['Fair','Good','Very Good','Premium','Ideal']])
diamondsDF["cut"] = ord_encoder.fit_transform(diamondsDF[['cut']])
hot_encoder = OneHotEncoder(handle_unknown='ignore')
diamonds_color_clarity = hot_encoder.fit_transform(diamondsDF[['color','clarity']]).toarray()
categories = []
for x in hot_encoder.categories_:
    categories.extend(x)
diamonds_color_clarity_df = pd.DataFrame(diamonds_color_clarity, columns=categories)
diamondsDF = diamondsDF.join(diamonds_color_clarity_df)
diamondsDF=diamondsDF.drop(['color','clarity'],axis=1)
data = diamondsDF.loc[:, diamondsDF.columns != "cut"]
y_data = diamondsDF[['cut']]
x_train, x_test = train_test_split(data, test_size=0.9, shuffle=True, random_state=100)
y_train, y_test = train_test_split(y_data, test_size=0.9, shuffle=True, random_state=100)

print(diamondsDF.shape)
scaler = StandardScaler() 
scaler.fit(x_train) 
x_train_scaled = scaler.transform(x_train)  
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled)


# In[1099]:


orig_r = km_em(x_train_scaled, dataset_name="Dataset 1", true_vals = y_train['cut'])


# In[1110]:


pca_result, x_projected_pca = pca_dim_red(x_train_scaled, dataset_name="Dataset 1", features_num = 15)
ica_result, x_projected_ica = ica_dim_red(x_train_scaled, dataset_name="Dataset 1", features_num = 12)
rp_result, x_projected_rp = rp_dim_red(x_train_scaled, dataset_name="Dataset 1", features_num = 15)
fa_result, x_projected_fa = fa_dim_red(x_train_scaled, dataset_name="Dataset 1", features_num = 12)


# In[1111]:


dataset_name = "Dataset 1"
n=22
pca_r = km_em(pca_result, dataset_name="Dataset 1 + PCA", true_vals = y_train['cut'])
ica_r = km_em(ica_result, dataset_name = "Dataset 1 + ICA", true_vals = y_train['cut'], reg_covar = 1e-06)
rp_r = km_em(rp_result, dataset_name = "Dataset 1 + Random Projection", true_vals = y_train['cut'])
fa_r = km_em(fa_result, dataset_name = "Dataset 1 + FeatureAgglomeration", true_vals = y_train['cut'])

plt.close()
plot_data(list(range(2, 23)), orig_r['sil'], title="K-means Silhouette Scores ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Silhouette Score", color="black", label='Orig')
plot_data(list(range(2, 23)), pca_r['sil'], title="K-means Silhouette Scores ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Silhouette Score", color="blue", label='PCA')
plot_data(list(range(2, 23)), ica_r['sil'], title="K-means Silhouette Scores ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Silhouette Score", color="orange", label='ICA')
plot_data(list(range(2, 23)), rp_r['sil'], title="K-means Silhouette Scores ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Silhouette Score", color="green", label='RP')
plot_data(list(range(2, 23)), fa_r['sil'], title="K-means Silhouette Scores ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Silhouette Score", color="red", label='FA')
plt.savefig((str(dataset_name))+' all silho.png')
plt.show()

plt.close()

plot_data(list(range(1, n)), orig_r['kmeans_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="black", label='Orig k-means')
plot_data(list(range(1, n)), orig_r['em_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="black", label='Orig EM', linestyle='dashed')

plot_data(list(range(1, n)), pca_r['kmeans_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="blue", label='PCA k-means')
plot_data(list(range(1, n)), pca_r['em_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="blue", label='PCA EM', linestyle='dashed')

plot_data(list(range(1, n)), ica_r['kmeans_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="orange", label='ICA k-means')
plot_data(list(range(1, n)),  ica_r['em_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="orange", label='ICA EM', linestyle='dashed')

plot_data(list(range(1, n)), rp_r['kmeans_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="green", label='RP k-means')
plot_data(list(range(1, n)),  rp_r['em_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="green", label='RP EM', linestyle='dashed')

plot_data(list(range(1, n)), fa_r['kmeans_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="red", label='FA k-means')
plot_data(list(range(1, n)),  fa_r['em_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="red", label='FA EM', linestyle='dashed')
plt.savefig((str(dataset_name))+' km-em all time.png')
plt.show()


# In[1116]:


plot_km_em_clusters(x_train_scaled, x=11, y=9, z = 5, dataset_name = "Dataset 1", km_clusters = 12, em_clusters=14)
plot_km_em_clusters(pca_result, x=12, y=7, z = 6, dataset_name = "Dataset 1 + PCA", km_clusters = 12, em_clusters=12)
plot_km_em_clusters(ica_result, x=11, y=7, z = 6, dataset_name = "Dataset 1 + ICA", km_clusters = 9, em_clusters=9)
plot_km_em_clusters(rp_result, x=11, y=9, z = 5, dataset_name = "Dataset 1 + Random Projection", km_clusters = 14, em_clusters=9)
plot_km_em_clusters(fa_result, x=11, y=6, z = 6, dataset_name = "Dataset 1 + FeatureAgglomeration", km_clusters = 7, em_clusters=7)
print(x_train_scaled.shape)
print(pca_result.shape)
print(ica_result.shape)
print(rp_result.shape)


# In[1119]:


#4

# MLPClassifier
plt.close()
mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 1", x_label="Training Size", y_label="Accuracy Score", color="black", label='Orig CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 1", x_label="Training Size", y_label="Accuracy Score", color="black", label='Orig Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_projected_pca, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 1", x_label="Training Size", y_label="Accuracy Score", color="blue", label='PCA CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 1", x_label="Training Size", y_label="Accuracy Score", color="blue", label='PCA Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_projected_ica, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 1", x_label="Training Size", y_label="Accuracy Score", color="red", label='ICA CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="(Neural Network Learning Curve - Dataset 1", x_label="Training Size", y_label="Accuracy Score", color="red", label='ICA Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_projected_rp, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 1", x_label="Training Size", y_label="Accuracy Score", color="orange", label='RP CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 1", x_label="Training Size", y_label="Accuracy Score", color="orange", label='RP Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_projected_fa, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 1", x_label="Training Size", y_label="Accuracy Score", color="green", label='FA CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 1", x_label="Training Size", y_label="Accuracy Score", color="green", label='FA Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

plt.savefig('Dataset 1 NN learning curve.png')
plt.show()


# In[1120]:


kmeans = KMeans(n_clusters=12, random_state=random_state)
y_pred = kmeans.fit_predict(x_train_scaled)
x_train_scaled_km = np.column_stack((x_train_scaled,y_pred))

plt.close()

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 1 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="black", label='Orig CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 1 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="black", label='Orig Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled_km, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 1 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="orange", label='CV (+K-means Result)', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 1 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training (+K-means Result')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

gm = BayesianGaussianMixture(n_components = 14, random_state=random_state, reg_covar=1e-01)
y_pred = gm.fit_predict(x_projected_pca)
x_train_scaled_em = np.column_stack((x_train_scaled,y_pred))


mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled_em, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 1 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="blue", label='CV (+EM Result)', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 1 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Training (+EM Result)')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

plt.savefig('Dataset 1 + Clustering NN learning curve.png')
plt.show()


# In[1122]:


# Dataset 2

wineDF = pd.read_csv('winequality-white.csv',sep=';')
data = wineDF.loc[:, wineDF.columns != "quality"]
y_data = wineDF[['quality']]
x_train, x_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=100)
y_train, y_test = train_test_split(y_data, test_size=0.2, shuffle=True, random_state=100)
scaler = StandardScaler()
scaler.fit(x_train) 
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[1123]:


print(wineDF)


# In[1125]:


orig_r = km_em(x_train_scaled, dataset_name="Dataset 2", true_vals = y_train['quality'])


# In[1126]:


pca_result, x_projected_pca = pca_dim_red(x_train_scaled, dataset_name="Dataset 2", features_num = 8)
ica_result, x_projected_ica = ica_dim_red(x_train_scaled, dataset_name="Dataset 2", features_num = 6)
rp_result, x_projected_rp = rp_dim_red(x_train_scaled, dataset_name="Dataset 2", features_num = 8)
fa_result, x_projected_fa = fa_dim_red(x_train_scaled, dataset_name="Dataset 2", features_num = 7)


# In[1127]:


dataset_name = "Dataset 2"
n=22
pca_r = km_em(pca_result, dataset_name="Dataset 2 + PCA", true_vals = y_train['quality'])
ica_r = km_em(ica_result, dataset_name = "Dataset 2 + ICA", true_vals = y_train['quality'], reg_covar = 1e-06)
rp_r = km_em(rp_result, dataset_name = "Dataset 2 + Random Projection", true_vals = y_train['quality'])
fa_r = km_em(fa_result, dataset_name = "Dataset 2 + FeatureAgglomeration", true_vals = y_train['quality'])

plt.close()
plot_data(list(range(2, 23)), pca_r['sil'], title="K-means Silhouette Scores ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Silhouette Score", color="blue", label='PCA')
plot_data(list(range(2, 23)), ica_r['sil'], title="K-means Silhouette Scores ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Silhouette Score", color="orange", label='ICA')
plot_data(list(range(2, 23)), rp_r['sil'], title="K-means Silhouette Scores ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Silhouette Score", color="green", label='RP')
plot_data(list(range(2, 23)), fa_r['sil'], title="K-means Silhouette Scores ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Silhouette Score", color="red", label='FA')
plt.savefig((str(dataset_name))+' all silho.png')
plt.show()

plt.close()

plot_data(list(range(1, n)), orig_r['kmeans_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="black", label='Orig k-means')
plot_data(list(range(1, n)), orig_r['em_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="black", label='Orig EM', linestyle='dashed')

plot_data(list(range(1, n)), pca_r['kmeans_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="blue", label='PCA k-means')
plot_data(list(range(1, n)), pca_r['em_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="blue", label='PCA EM', linestyle='dashed')

plot_data(list(range(1, n)), ica_r['kmeans_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="orange", label='ICA k-means')
plot_data(list(range(1, n)),  ica_r['em_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="orange", label='ICA EM', linestyle='dashed')

plot_data(list(range(1, n)), rp_r['kmeans_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="green", label='RP k-means')
plot_data(list(range(1, n)),  rp_r['em_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="green", label='RP EM', linestyle='dashed')

plot_data(list(range(1, n)), fa_r['kmeans_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="red", label='FA k-means')
plot_data(list(range(1, n)),  fa_r['em_times'], title="k-means/EM Running Time ("+(str(dataset_name))+")", x_label="Number of Clusters", y_label="Time", color="red", label='FA EM', linestyle='dashed')
plt.savefig((str(dataset_name))+' km-em all time.png')
plt.show()


# In[1128]:


def graph_metric(attr='homogeneity', attr2='homogeneity_em',title="Homogeneity (Dataset 2)", clusters = 7):
    labels = ['Orig', 'PCA', 'ICA', 'RP', 'FA']
    attr1_means = [orig_r[attr][clusters-1], pca_r[attr][clusters-1], ica_r[attr][clusters-1],rp_r[attr][clusters-1], fa_r[attr][clusters-1]]
    attr2_means = [orig_r[attr2][clusters-1], pca_r[attr2][clusters-1], ica_r[attr2][clusters-1],rp_r[attr2][clusters-1], fa_r[attr2][clusters-1]]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, attr1_means, width, label='K-means', color='orange')
    rects2 = ax.bar(x + width/2, attr2_means, width, label='EM', color='lightblue')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig(title+'.png')
    plt.show()
graph_metric(attr='homogeneity', attr2='homogeneity_em',title="Homogeneity (Dataset 2 - 7 Clusters)", clusters = 7)
graph_metric(attr='completeness', attr2='completeness_em',title="Completeness (Dataset 2 - 7 Clusters)", clusters = 7)
graph_metric(attr='mutual_info', attr2='mutual_info_em',title="Mutual Information (Dataset 2 - 7 Clusters)", clusters = 7)
graph_metric(attr='adj_rand_score', attr2='adj_rand_score_em',title="Adjusted Rand Score (Dataset 2 - 7 Clusters)", clusters = 7)


# In[1129]:


plot_km_em_clusters(x_train_scaled, x=0, y=2, z = 6, dataset_name = "Dataset 2", km_clusters = 7, em_clusters=7)
plot_km_em_clusters(x_projected_pca, x=0, y=2, z = 6, dataset_name = "Dataset 2 + PCA", km_clusters = 7, em_clusters=7)
plot_km_em_clusters(x_projected_ica, x=0, y=2, z = 6, dataset_name = "Dataset 2 + ICA", km_clusters = 7, em_clusters=7)
plot_km_em_clusters(x_projected_rp, x=0, y=2, z = 6, dataset_name = "Dataset 2 + Random Projection", km_clusters = 7, em_clusters=7)
plot_km_em_clusters(x_projected_fa, x=0, y=2, z = 6, dataset_name = "Dataset 2 + FeatureAgglomeration", km_clusters = 7, em_clusters=7)

print(x_train_scaled.shape)
print(pca_result.shape)
print(ica_result.shape)
print(rp_result.shape)


# In[1130]:


#4

# MLPClassifier
plt.close()
mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 2", x_label="Training Size", y_label="Accuracy Score", color="black", label='Orig CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 2", x_label="Training Size", y_label="Accuracy Score", color="black", label='Orig Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_projected_pca, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 2", x_label="Training Size", y_label="Accuracy Score", color="blue", label='PCA CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 2", x_label="Training Size", y_label="Accuracy Score", color="blue", label='PCA Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_projected_ica, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 2", x_label="Training Size", y_label="Accuracy Score", color="red", label='ICA CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="(Neural Network Learning Curve - Dataset 2", x_label="Training Size", y_label="Accuracy Score", color="red", label='ICA Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_projected_rp, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 2", x_label="Training Size", y_label="Accuracy Score", color="orange", label='RP CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 2", x_label="Training Size", y_label="Accuracy Score", color="orange", label='RP Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_projected_fa, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 2", x_label="Training Size", y_label="Accuracy Score", color="green", label='FA CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 2", x_label="Training Size", y_label="Accuracy Score", color="green", label='FA Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

plt.savefig('Dataset 2 NN learning curve.png')
plt.show()


# In[1131]:


kmeans = KMeans(n_clusters=9, random_state=random_state)
y_pred = kmeans.fit_predict(x_train_scaled)
x_train_scaled_km = np.column_stack((x_train_scaled,y_pred))

plt.close()

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 1 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="black", label='Orig CV', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 1 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="black", label='Orig Training')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled_km, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 2 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="orange", label='CV (+K-means Result)', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 2 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="orange", label='Training (+K-means Result')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

gm = BayesianGaussianMixture(n_components = 9, random_state=random_state, reg_covar=1e-01)
y_pred = gm.fit_predict(x_train_scaled)
x_train_scaled_em = np.column_stack((x_train_scaled,y_pred))


mlp_learner = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.07)
train_sizes, train_scores, test_scores = learning_curve(mlp_learner, x_train_scaled_em, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plot_data(train_sizes, test_mean, title="Neural Network Learning Curve - Dataset 2 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="blue", label='CV (+EM Result)', linestyle='dashed')
plot_data(train_sizes, train_mean, title="Neural Network Learning Curve - Dataset 2 + Clustering Result", x_label="Training Size", y_label="Accuracy Score", color="blue", label='Training (+EM Result)')
print(train_sizes)
print((train_sizes[np.argmax(test_mean)]))
print(test_mean[np.argmax(test_mean)])
print(train_mean[np.argmax(test_mean)])

plt.savefig('Dataset 2 + Clustering NN learning curve.png')
plt.show()

