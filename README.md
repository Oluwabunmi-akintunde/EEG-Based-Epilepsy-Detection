# EEG-Based-Epilepsy-Detection
Machine learning python script for epilepsy detection using EEG data
# Epilepsy Detection Using EEG Features – Jupyter Notebook

This repository contains a Jupyter Notebook that walks through the complete pipeline for detecting epilepsy using machine learning models on EEG-features.

## Notebook Overview

###  'Math516 Coursework - EEG-Based Epilepsy Detection.ipynb'
This notebook contains all the analysis, broken into six clearly labeled sections:

1. **Data Preparation**
   - Loads and explores the EEG dataset
   - Sets up target labels

2. **Feature Analysis and Dimensionality Reduction**
   - 2.1 Removes features with missing values above 75%
   - 2.2 Removes low- and no-variance features
   - 2.3 Identifies highly correlated (redundant) features
   - 2.4 Compiles all features to drop based on the above to reduce dimensionality

3. **Model Selection**
   - Trains 11 machine learning models using cross-validation
   - Ranks models by mean accuracy
   - Selects top 5 classifiers (SVM, Random Forest, Extra Trees, Bagging, Logistic Regression)

4. **Feature Selection**
   - Applies Recursive Feature Elimination with Cross-Validation (RFECV)
   - Identifies most informative EEG features for classification

5. **Model Training Using All 36 Features**
   - Trains and evaluates the top 5 models using the full filtered feature set
   - Calculates performance metrics: Accuracy, Sensitivity, Specificity, AUC, Precision, F1 Score

6. **Model Training Using Selected Features**
   - Trains and evaluates the top 5 models using the reduced feature set from RFECV
   - Compares performance to assess the benefit of dimensionality reduction

## Outputs

- ROC curves and confusion matrices for all models
- Performance metrics summary table
- Barplots showing top and bottom feature importance
- Correlation matrix for initial 40 EEG features
- Boxplot for training the 11 models using cross-validation

## Requirements

- Jupyter Notebook
- scikit-learn
- pandas
- seaborn
- matplotlib
- numpy

## Author

Oluwabunmi Akintunde
