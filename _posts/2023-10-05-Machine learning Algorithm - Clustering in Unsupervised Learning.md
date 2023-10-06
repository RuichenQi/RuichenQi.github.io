---
layout: post
title: Machine learning Algorithm - Clustering in Unsupervised Learning
subtitle: This article introduces the concept of unsupervised learning, K-means algorithm and how to choose number of clusters.
categories: AI
tags: Machine_Learning_Algorithms
---

## Concept of unsupervised learning
Unsupervised learning is a paradigm in machine learning where, in contrast to supervised learning and semi-supervised learning, algorithms learn patterns exclusively from unlabeled data, that is to say, the algorithms work with data that doesn't have any predefined labels or target values. Unsupervised learning operates purely on the input data. The goal of unsupervised learning is to discover hidden patterns or structures within the data. These patterns could be clusters of similar data points, relationships between features, or underlying distributions.

## Clustering: K-means algorithm
### Steps of K-means algorithm
The K-means algorithm is the most popular used algorithm for grouping data into coherent subsets. In this example, our goal is to separate some points on a coordinate plane into k groups.

Steps of K-means algorithm is as follows:

1. Randomly initialize k points in the dataset as the cluster centroids.
2. Assign all examples into k clusters based on which cluster centroid the example is closest to.
3. Compute the averages for all the points inside each of the k cluster centroid groups, then move the cluster centroid points to those averages.
4. Repeat step 2 and step 3 until we have found our clusters.

It is worth mentioning that Some datasets have no real inner separation or natural structure. K-means can still evenly segment 
your data into K subsets, so can still be useful.

### Optimization objective of clustering

![clustering_optimize](https://ruichenqi.github.io/assets/images/AI/2/clustering_optimize.png)

It is clear that in step 2, our goal is to minimize cost function J() with c (holding μ fixed), while in step 3, our goal is to minimize cost function J() with μ.

With K-means algorithm, it is not possible for the cost function to increase. It should always descend.

## Choosing number of clusters

We can use the elbow method to determine which number of clusters we should use. That is, plot the cost function J() in corresponding to the number of cluster k. When we increase k, the cost J will reduce, and then flatten out. Choose the point when k starting to flatten out as our number of clusters.

Sometimes you are running K-means algorithm for some downstream purpose. In this situation, evaluate K-means based on a metric for how it will performs for that later purpose.