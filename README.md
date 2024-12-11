# wine-classification-svm
This repository contains a Python implementation of a Support Vector Machine (SVM) classifier for predicting wine cultivators based on chemical analysis. The dataset used is the Wine dataset from `sklearn.datasets`.

## Overview

The Wine dataset contains chemical analysis of 178 wine samples derived from three different cultivators. This project demonstrates the process of using SVMs for multiclass classification with the following steps:
- Data loading and preprocessing
- Feature scaling using `StandardScaler`
- Handling multiclass classification with a one-vs-rest strategy
- Evaluating performance with cross-validation
- Hyperparameter tuning for optimal performance

## Steps

1. **Load the Dataset**: The Wine dataset is loaded using `sklearn.datasets.load_wine`.
2. **Data Splitting**: The data is split into training and test sets.
3. **Feature Scaling**: `StandardScaler` is used to normalize the features.
4. **Linear SVM**: A simple linear SVM (`LinearSVC`) is implemented and evaluated.
5. **Kernelized SVM**: A kernelized SVM (`SVC`) is trained, and hyperparameter tuning is performed using `RandomizedSearchCV`.
6. **Model Evaluation**: The best model is tested on the test set for final performance evaluation.

## Key Results

- The linear SVM achieved reasonable accuracy after scaling the features.
- Kernelized SVM with hyperparameter tuning significantly improved performance.
- The model achieved a high accuracy score on the test set.
