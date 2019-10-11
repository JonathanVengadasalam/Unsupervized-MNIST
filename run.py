# -*- coding: utf-8 -*-
%matplotlib inline
%config IPCompleter.greedy=True
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
from math import ceil

from itertools import combinations, product
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")
x_not_scaled = mnist.data[::50, :].astype(float)
y = mnist.target[::50]
x_not_scaled.shape

data = np.concatenate( (x_not_scaled,np.reshape(y,(y.shape[0],1))), axis=1)
np.random.shuffle(data)
data[0:10,784]

data = np.concatenate( (x_not_scaled,np.reshape(y,(y.shape[0],1))), axis=1 )
np.random.shuffle(data)
x_not_scaled = data[:,0:-1]
y = data[:,-1]

from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler().fit(x_not_scaled)
x = std_scale.transform(x_not_scaled)

plt.figure(figsize=(5,5))
plt.hist(y, bins=30, color="steelblue")
plt.xticks(range(0,10))
plt.axis([-0.5,9.5,110,170])
plt.title("Y distribution", fontsize=14);

from sklearn.decomposition import KernelPCA

pca = KernelPCA(n_components=4, kernel="rbf")
xprojected = pca.fit_transform(x)

yColor_on_xProjection(xprojected, y, 2)

pca = KernelPCA(n_components=2, kernel="cosine")
pca.fit(x)
xprojected = pca.transform(x)

yColor_xImshow_on_xProjection( xprojected, y, x_not_scaled )

from sklearn.manifold import TSNE
xprojected_tsne = TSNE(n_components=2,\
                  perplexity=30,\
                  learning_rate=200,\
                  n_iter=5000)\
            .fit_transform(x)
yColor_xImshow_on_xProjection( xprojected_tsne, y, x_not_scaled )

from sklearn.cluster import KMeans
km = KMeans(n_clusters=10, init="k-means++", max_iter=500, n_init=20, n_jobs=2)
km.fit(x_not_scaled)
matrix_heatmap(confusion_matrix(y, km.labels_), title="Distribution des clusters sur le target")

plt_clusters_shape(x_not_scaled, km.labels_)

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=10, affinity="euclidean", compute_full_tree=True, linkage="ward")
ac.fit(x_not_scaled)
matrix_heatmap(confusion_matrix(y, ac.labels_), title="Distribution des clusters sur le target")

plt_clusters_shape(x_not_scaled, ac.labels_)

random_sample(x_not_scaled[ac.labels_ == 3], 40)

silhouette_score(x_not_scaled, ac.labels_)

yColor_xImshow_on_xProjection( xprojected_tsne, y, y2color=ac.labels_ )
plt.title("Target (number) vs Cluster (color) on TSNE projection", fontdict={"fontsize":"20"});

print("adjusted rand : {}\nsilhouette : {}\naccuracy : {}"\
      .format(adjusted_rand_score(y, km.labels_),\
              silhouette_score(x_not_scaled, km.labels_),\
              accuracy_score(y, convert_cluster_labels(km.labels_, [1,7,8,3,0,4,6,2,0,1]))))
              
print("adjusted rand : {}\nsilhouette : {}\naccuracy : {}"\
      .format(adjusted_rand_score(y, ac.labels_),\
              silhouette_score(x_not_scaled, ac.labels_),\
              accuracy_score(y, convert_cluster_labels(ac.labels_, [0,8,7,3,6,9,2,1,1,3]))))
