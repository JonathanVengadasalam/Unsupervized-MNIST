# Unsupervized-MNIST
An unsupervized machine learning with Python [Scikit-Learn](https://scikit-learn.org/stable/) on [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset

## About The Project
This project is an overview of principal unsupervized methods on mnist datasets :
 - principal component analysis
 - t-distributed stochastic neighbor embedding
 - k-means clustering
 - agglomerative clustering

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

### Displays and Results
An random sample of dataset :
![random sample of x](https://github.com/JonathanVengadasalam/Unsupervized-MNIST/blob/master/images/x%20random%20sample.png)
PCA with radial basis function kernel (rbf) for the 4 first components. The color determinesthe target y (0 to 9) on the components projections.
![pca rbf](https://github.com/JonathanVengadasalam/Unsupervized-MNIST/blob/master/images/pca%20(kernel%3Drbf).png)
## Acknowledgements
* [OpenClassrooms](https://openclassrooms.com/fr/courses/4379436-explorez-vos-donnees-avec-des-algorithmes-non-supervises)

