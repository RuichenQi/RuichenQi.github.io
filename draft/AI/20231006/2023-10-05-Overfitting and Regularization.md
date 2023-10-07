---
layout: post
title: Overfitting and Regularization
subtitle: 
categories: AI
tags: Machine_Learning_Algorithms
---

## What is a good model?

In the previous note, the definition of a 'good' model is 
the one with least training error ( output of cost function ). In the extreme case, this ‘good’ model has zero training error and fits training data perfectly, that is, the predict label of all training data is equal to their corresponding truth label. However, sometimes we may find that our trained model can work perfectly on training set ( the least training error ) while it can not performance well on the testing set. So what happened?

## Underfit and overfit

Underfitting means that the model has not learned the training data well enough, and has failed to capture the underlying relationship between the input and output variables. This makes the model perform poorly on both the training data and new data. Underfitting is often caused by using a too simple model, having too few features, or having too much regularization.

Overfitting means that the model has learned the training data too well, and has memorized the specific details and noise of the data, rather than the general pattern. This makes the model perform very well on the training data, but poorly on new data that may have different or unseen variations. Overfitting is often caused by using a too complex model, having too many features, or not having enough training data.

Here is an illustration showing the concept of overfitting:

![Overfitting](https://ruichenqi.github.io/assets/images/AI/2/overfitting.png)

As illustrated, we want our model to fit red points ( the training set ). In this example, our model is trained and has a small training error. However, when the new data comes ( the green points), our model can not work well. In other words, our model is 'overfitting'.

## How to judge whether the model is underfitting or overfitting

The learned model works terrible for both training data and testing data - the model is underfitting.

The learned model works well for training data but 
terrible for testing data - the model is overfitting.

## Ways to addressing overfitting
- Reduce number of features ( lose some information ).
  - Manually select which features to keep.
  - Use dimensionality reduction algorithms.
- Try other algorithms.
- Regularization
  - Keep all features, but reduce magnitude of parameters θ.
  - Works well when we have a lot of features, each of which contributes a bit to predicting y.

## Regularization



