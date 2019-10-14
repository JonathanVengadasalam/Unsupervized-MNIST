# Unsupervized-MNIST
An unsupervized machine learning with Python [Scikit-Learn](https://scikit-learn.org/stable/) on [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset

## About The Project
This project is an overview of principal unsupervized methods on mnist datasets :
 - principal component analysis (pca)
 - t-distributed stochastic neighbor embedding (tsne)
 - k-means clustering (km)
 - agglomerative clustering (ac)

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

## Displays and Results
An random sample of dataset.
![random sample of x](https://github.com/JonathanVengadasalam/Unsupervized-MNIST/blob/master/images/x%20random%20sample.png)

PCA with radial basis function kernel (rbf) for the 4 first components. The color represents the target y (0 to 9) on the components projections.
![pca rbf](https://github.com/JonathanVengadasalam/Unsupervized-MNIST/blob/master/images/pca%20(kernel%3Drbf).png)

PCA with cosine kernel for the 2 first components. The image, color and number represents the target y on the components projections.
![pca cosine](https://github.com/JonathanVengadasalam/Unsupervized-MNIST/blob/master/images/pca%20(kernel%3Dcosine).png)

TSNE on 2 components, perplexity=30, learning_rate=200. The image, color and number represents the target y on the components projections. ![tsne](https://github.com/JonathanVengadasalam/Unsupervized-MNIST/blob/master/images/tsne%20(perplexity%3D10%20-%20epsilon%3D200).png)

K-Means clustering's confusion matrix.

![k-means](https://github.com/JonathanVengadasalam/Unsupervized-MNIST/blob/master/images/km%20distributions.png)

k-Means clustering shape.

![k-means clusters shape](https://github.com/JonathanVengadasalam/Unsupervized-MNIST/blob/master/images/k-means%20clusters%20shape.png)

## Acknowledgements
* [OpenClassrooms](https://openclassrooms.com/fr/courses/4379436-explorez-vos-donnees-avec-des-algorithmes-non-supervises)

