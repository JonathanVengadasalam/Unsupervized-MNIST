# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from itertools import combinations, product
from math import ceil

def plt_random_sample(x, n_sample=20):
    
    select = x[np.random.randint(x.shape[0], size=n_sample)]
    n_line = ceil(n_sample/20)
    fig, axes = plt.subplots(n_line, 20, figsize=(20, 1.4*n_line))
    
    for ax, i in zip(axes.ravel(),range(select.shape[0])):
        ax.imshow(select[i, :], cmap="binary")
        ax.set_xticks([]), ax.set_yticks([])

def plt_clusters_shape(x, y):
    tmp = []
    
    for i in range(10):
        tmp.append(np.mean(x_not_scaled[y == i], axis=0))
    tmp = np.array(tmp)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    
    for ax, i in zip(axes.ravel(),range(tmp.shape[0])):
        ax.imshow(np.reshape(tmp[i, :], (28, 28)), cmap="binary")
        ax.set_title("Cluster : {}".format(i))
        ax.set_xticks([]), ax.set_yticks([])
        
def plt_2axes_projection(xprojected, y2text, x2imshow=None, y2color=None, ax=None, figsize=(20,10), ysize=16):
    xmin, xmax = np.min(xprojected,0), np.max(xprojected,0)
    x = (xprojected -xmin) / (xmax - xmin)
    if type(y2color) != np.ndarray:
        y2color = y2text
    shown_images = np.array([[1.,1.]])
    
    if ax == None:
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    
    for i in range(x.shape[0]):
        ax.text(x[i,0], x[i,1], str(int(y2text[i])), color=plt.cm.Set2(y2color[i]/10.),\
                fontsize = ysize)
    
    if type(x2imshow) == np.ndarray:
        for i in range(x.shape[0]):
            if np.min(np.sum((shown_images - x[i])**2, 1)) > 4e-3:
                shown_images = np.r_[shown_images, [x[i]]]
                ax.add_artist(AnnotationBbox(OffsetImage(np.reshape(x2imshow[i, :],(28, 28)),cmap=plt.cm.gray_r,zoom=0.5), x[i]))
    plt.xticks([]), plt.yticks([])
    
def plt_projections(x, y, size=1):
    combs = list(combinations(range(x.shape[1]), 2))
    n_line = ceil(len(combs)/3)
    fig, axes = plt.subplots(n_line, 3, figsize=(18, 5*n_line))

    for ax, comb in zip(axes.ravel(), combs):
        scatter = ax.scatter(x[:,comb[0]], x[:,comb[1]], c=y, s=size)
        ax.set_title("Components : {} - {}".format(comb[0]+1,comb[1]+1))
        
    fig.colorbar(scatter, ax=axes, orientation="horizontal", fraction=0.05/n_line, pad=0.1/n_line)
    
def plt_matrix_heatmap(cm, normalize=False, figsize=(5,5), title="Heatmap", cmap=plt.cm.Blues):
    thresh = cm.max() / 2.
    fmt = "d"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    
    fig = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.xticks(range(10)), plt.yticks(range(10))
    plt.xlabel("Cluster values"), plt.ylabel("Target values")
    plt.title(title)
    
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
