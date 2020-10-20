# ImageCluster

## Overview

This is a quick-and-dirty attempt at image clustering and image retrieval. 
When working with images one must carefully consider feature representation. 
But what constitues a *feature* in an image? Is it a pixel, a superpixel, 
a corner, a vertical edge, a horizontal edge, gradient, etc? In NLP the task 
of feature extraction is relatively easy - features are typically BoW (bag-of-
words), and/or linguistic structure such as POS (part-of-speech) tags, or as 
is very common nowadays, word or document embeddings (word2vec, GloVe).

Since I have no experience in image processing, and very limited time, I opted 
for a pre-trained convolutional neural network (Simonyan and Zisserman 2014) as 
my feature representation. I use the output of the final pooling layer as my 
image feature vector. The complete flow can be seen as follows:

1. Use convnet VGG16 to create an image embedding
2. Project the image embedding onto latent space - in this case we do this via 
Principal Component Analysis (PCA)
3. Run the K-means clustering algorithm on the PCA vectors to find cluster 
centroids; we use the "elbow"/"knee" method to find optimal number of clusters 
*k*

In general this works quite well - plotting the clusters and eyeballing the 
clustered images seems to confirm this. However it is by no means perfect.

## Pre-requisites

- Python 2.8
- See `requirements.txt` for complete list of modules

## Usage

Simply run `ImageCluster.ipynb` notebook. Note that I've created some helper 
classes in `data.py`, `features.py` and `preprocess.py`.

## Future Work

Now that we have a baseline with convnet, obvious future work would involve 
using more traditional image processing to extract feature vectors. Some 
possibilities include:

- Superpixel segments (e.g. using Simple Linear Iterative Clustering, SLIC)
- Haar-like features
- Histogram of oriented gradients (HOG)

A very interesting paper I came across and would like to replicate is Latent 
Dirichlet Allocation (LDA) for images (Elango and Jayaraman 2005). Since I've 
applied LDA in NLP tasks before, I could try to replicate the process for 
images, using one (or combination) of the features mentioned above.

## References

*Simonyan, K.; and Zisserman, A. 2014, Very Deep Convolutional Networks for Large-Scale Image Recognition*

*Elango, P. K.; and Jayaraman, K. 2005, Clustering Images Using the Latent Dirichlet Allocation Model*