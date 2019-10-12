# -*- coding: utf-8 -*-
%matplotlib inline
%config IPCompleter.greedy=True
import numpy as np
from utils import *

#import data
from tensorflow.python.keras.datasets import mnist
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xdata = np.concatenate((xtrain,xtest))
ydata = np.concatenate((ytrain,ytest))
x_notscaled = xdata[::50,:,:]
y = ydata[::50]

#shuffle data
from sklearn.utils import shuffle
x_notscaled, y = shuffle(x_notscaled, y)
random_sample(x_notscaled,40)

#scale data
from sklearn.preprocessing import StandardScaler
x_reshaped = np.reshape(x_notscaled,(x_notscaled.shape[0],-1))
std_scale = StandardScaler().fit(x_reshaped)
x = std_scale.transform(x_reshaped)
x = x.reshape((x.shape[0],28,28))


from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components=4, kernel="rbf")
xprojected = pca.fit_transform(x)
yColor_on_xProjection(xprojected, y, 2)

pca = KernelPCA(n_components=2, kernel="cosine")
pca.fit(x)
xprojected = pca.transform(x)
yColor_xImshow_on_xProjection( xprojected, y, x_not_scaled )

from sklearn.manifold import TSNE
xprojected_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=5000).fit_transform(x)
yColor_xImshow_on_xProjection( xprojected_tsne, y, x_not_scaled )

from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

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
