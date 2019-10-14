# -*- coding: utf-8 -*-
%matplotlib inline
%config IPCompleter.greedy=True
import numpy as np
from plot_functions import *

from tensorflow.python.keras.datasets import mnist
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

def import_data():
    #import data
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    xdata = np.concatenate((xtrain,xtest))
    ydata = np.concatenate((ytrain,ytest))
    x_notscaled = xdata[::50,:,:]
    x_notscaled = np.reshape(x_notscaled,(x_notscaled.shape[0],-1))
    y = ydata[::50]

    #shuffle data
    x_notscaled, y = shuffle(x_notscaled, y)
    plt_random_sample(x_notscaled,40)

    #scale data
    std_scale = StandardScaler().fit(x_notscaled)
    x = std_scale.transform(x_notscaled)
    
    return x_notscaled, x, y

def pca(x, y, x_notscaled=None):
    #pca kernel=rbf
    xprojected = KernelPCA(n_components=4, kernel="rbf").fit_transform(x)
    plt_projections(xprojected, y, 2)

    #pca kernel=cosine
    xprojected = KernelPCA(n_components=2, kernel="cosine").fit_transform(x)
    if type(x_notscaled) == type(None): x_notscaled = x
    plt_2axes_projection(xprojected, y, x_notscaled)

def tsne(x, y, x_notscaled=None):
    xprojected_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=5000).fit_transform(x)
    if type(x_notscaled) == type(None): x_notscaled = x
    plt_2axes_projection(xprojected_tsne, y, x_notscaled)
    return xprojected_tsne

def km_clustering(x, y):
    km = KMeans(n_clusters=10, init="k-means++", max_iter=500, n_init=20, n_jobs=2)
    km.fit(x)
    plt_matrix_heatmap(confusion_matrix(y, km.labels_), title="clusters and tagets distribution")
    plt_clusters_shape(x, km.labels_)
    print("adjusted rand : {}\nsilhouette : {}"\
          .format(adjusted_rand_score(y, km.labels_),silhouette_score(x_notscaled, km.labels_)))

def ac_clustering(x, y, projected=None):
    ac = AgglomerativeClustering(n_clusters=10, affinity="euclidean", compute_full_tree=True, linkage="ward")
    ac.fit(x)
    matrix_heatmap(confusion_matrix(y, ac.labels_), title="clusters and tagets distribution")
    plt_clusters_shape(x, ac.labels_)
    random_sample(x[ac.labels_ == 3], 40)
    if type(projected) != type(None):
        plt_2axes_projection(projected, y, y2color=ac.labels_)
        plt.title("Target (number) vs Cluster (color) on TSNE projection", fontdict={"fontsize":"20"})
    print("adjusted rand : {}\nsilhouette : {}"\
          .format(adjusted_rand_score(y, ac.labels_),silhouette_score(x_notscaled, ac.labels_)))             
