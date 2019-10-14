# Unsupervized-MNIST
An unsupervized machine learning with Python [Scikit-Learn](https://scikit-learn.org/stable/) on [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset

## Table of Contents
* [About the Project](#about-the-project)
* [Acknowledgements](#acknowledgements)

## About The Project
This project 

## Getting Started
### Prerequisites
* numpy `py -m pip install numpy`
* scikitlearn `pip install -U scikit-learn`

### Usage
1. `from functions import *`
2. unzip the mnist dataset in the folder data `unzip()`
3. load dataset `x_notscaled, x, y = load_data()`
4. plot the different unsupevized functions and get the results
   ```sh
   pca(x, y, x_notscaled)
   projected = tsne(x, y, x_notscaled)
   km_clustering(x, y)
   ac_clustering(x, y, projected)
   ```
## Acknowledgements
* [OpenClassrooms](https://openclassrooms.com/fr/courses/4379436-explorez-vos-donnees-avec-des-algorithmes-non-supervises)

