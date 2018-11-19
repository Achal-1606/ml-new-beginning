-------------------------------
## Some important terminology
-------------------------------

* Covariance Vs Correlation
    - Both describe the relationship between 2 variables
    - __Covariance__ - It is defined as a systematic relationship between a pair of random variable, where in changes in one variable reciprocted by change an equivalent change in another variable.
    - __Correlation__ - It determines the degree to which two or more random variables are related. 
    - Correlation is when the change in one item may result in the change in another item. While covariance is when two items vary together.
    - Important Differences
        + A measure used to indicate the extend to which two random variables change in tandem is known as covariance. A measure used to represent how strongly two random variables are related is known as correlation.
        + __Covariance is nothing but a measure of correlation. On the contrary, correlation refers to the scaled form of covariance.__
        + The value of correlation takes place between -1 and +1. Conversely, the value of covariance lies between -∞ and +∞.
        + Covariance is affected by the change in scale, and correlation is not affected by change in scale.
        + __Correlation is a special case of covariance which can be obtained when the data is standardized.__
        + Correlation is preffered in decision making, because it remains unaffected by location and scale.


* Coeffecient of Variation (CV)
    - It is known as Relative Standard Deviation. basically the __ratio of standard deviation to mean.__
    - It is a standardized measure of dispersion of a probability distribution or frequency distribution
    - The scattered frequency distribution will have a higher CV and viceversa.

* Variance
    - Measure of variability / __how spread the data is?__
    - __Average square deviation from the Mean__


-------------------------------
## Principal Component Analysis
-------------------------------

* used for dimensionality reduction
* PCA consist of 3 steps
    - Compute the co-varianvce matrix
    - Compute the Eigen values and vectors of this covariance matrix
    - Use the eigen values and vectors select only the most important vectors and transform the data onto those vectors for reduced dimentionality


#### 1. Compute the Covariance Matrix
* PCA yields a feature subspace that __maximize variance along the feature vectors.__
* For measuring variance, as mentioned above the __data need to be normalized to have zero-mean and unit-variance__, such that each feature is weighed equally in out calculations. <br>
```from sklearn.preprocessing import StandardScaler```<br>
```X = fit_transform(X)```<br>
* ###### Covariance Formula -
![covariance formula](https://cdn-images-1.medium.com/max/800/1*kWwfHg0cbL1-4_8yX0VhlA.png "covariance formula")<br>
Where the x with the line on top is a vector of mean values for each feature of X. Notice that when we multiply a transposed matrix by the original one we end up multiplying each of the features for each data point together!

* Implementation in Python -
```

import numpy as np

# Compute the mean of the data
mean_vec = np.mean(X, axis=0)

# Compute the covariance matrix
cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0]-1)


# OR we can do this with one line of numpy:
cov_mat = np.cov(X.T)
```


#### 2. Compute Eigen values and vectors
* Eigen Vector - Vector direction of new feature space
* Eigen Value - magnitude of those vectors
* __Higher Eigen Value__ --> result in __higher variance__ --> hold a __lot of information__ about our data
* On the contrary, __Lower Eigen Value__ --> result in __lower variance__ --> hold __less information__ about our data
* The whole idea is to find the vectors that are the most important in representing our data and discard the rest. In Python -
```
# Compute the eigen values and vectors using numpy
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
```

#### 3. Compute Eigen values and vectors
* We want to select the most important feature vectors that we need and discard the rest. 
* We can do this in a clever way by looking at the explained variance percentage of the vectors. This percentage quantifies how much information (variance) can be attributed to each of the principal components out of the total 100%.
* In the code below, we simply count the number of feature vectors we would like to keep based on a selected threshold of 97%.
```
# Only keep a certain number of eigen vectors based on 
# the "explained variance percentage" which tells us how 
# much information (variance) can be attributed to each 
# of the principal components

exp_var_percentage = 0.97 # Threshold of 97% explained variance

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

num_vec_to_keep = 0

for index, percentage in enumerate(cum_var_exp):
  if percentage > exp_var_percentage:
    num_vec_to_keep = index + 1
    break
```
* Project the data on the selected Eigen vectors
```
# Compute the projection matrix based on the top eigen vectors
num_features = X.shape[1]
proj_mat = eig_pairs[0][1].reshape(num_features,1)
for eig_vec_idx in range(1, num_vec_to_keep):
  proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))

# Project the data 
pca_data = X.dot(proj_mat)
```
Source:- [Blog Post link for above Content](https://towardsdatascience.com/principal-component-analysis-your-tutorial-and-code-9719d3d3f376)

-------------------------------
## PCA (Explaned More)
-------------------------------
* Data Analysis Requirement(find pattern in data) require data to be -
    - spread across each dimensions
    - dimensions to be independent
* What PCA does?
    - PCA __finds a new set of dimensions__ (or a set of basis of views) such that all the __dimensions are orthogonal__ (and hence linearly independent) and __ranked according to the variance of data__ along them. It means more important principle axis occurs first. (more important = more variance/more spread out data)
* Covariance Matrix properties<br>
!["Covariance Matrix"](https://cdn-images-1.medium.com/max/600/1*28OtA0VsUZiXyYT5AhKVoA.png "Covariance Matrix")<br>
A covariance matrix of some data set in 4 dimensions a,b,c,d. 
Va : variance along dimension a
Ca,b : Covariance along dimension a and b <br>

    - __variance__ of dimensions as the __main diagonal elements__
    - __covariance__ of dimensions as the __off diagonal elements__

* Now for PCA we require...
    - data to be spread out i.e. it should have high variance along dimensions.
    - want to remove correlated dimensions i.e. covariance among the dimensions should be zero
    - Therefore, __our covariance matrix should have__ -
        + large numbers as the main diagonal elements.
        + zero values as the off diagonal elements.
* Obtaining New Data <br>
![Transforming Data](https://cdn-images-1.medium.com/max/800/1*i-oS46CO9S67LP2V0sAuog.png "Transforming Data")<br>

Source - [Blog link for the above content](https://medium.com/@aptrishu/understanding-principle-component-analysis-e32be0253ef0)


