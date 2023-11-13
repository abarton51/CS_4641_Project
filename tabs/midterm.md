---
layout: default
title: 2. Midterm Report
---
# CS 4641 Machine Learning, Team 6, Midterm Report
Members: Austin Barton, Karpagam Karthikeyan, Keyang Lu, Isabelle Murray, Aditya Radhakrishan

## Introduction
A quick introduction of your topic and mostly literature review of what has been done in this area. Briefly explain your dataset and its features and provide a link to your dataset if applicable.

### Datasets
**MusicNet**: Description and link to dataset.
MusicNet is a collection of 330 freely-licensed classical music recordings, together with over 1 million annotated labels indicating the precise time of each note in every recording, the instrument that plays each note, and the note's position in the metrical structure of the composition. The labels are acquired from musical scores aligned to recordings by dynamic time warping. The labels are verified by trained musicians; a labeling error rate of 4% has been estimated. The MusicNet labels are offered to the machine learning and music communities as a resource for training models and a common benchmark for comparing results.
https://www.kaggle.com/datasets/imsparsh/musicnet-dataset 

**GTZAN**: Description and link to dataset.
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification 

## Problem Definition
Why there is a problem here? What is the motivation of the project? 

## Methods

### Exploratory Data Analysis (EDA)
**MusicNet**:

**GTZAN**:

### Data Preprocessing
**MusicNet**:

**GTZAN**:

#### Dimensionality Reduction - PCA
**MusicNet**:
- Brief description and include visuals if applicable.

**GTZAN**:
- Brief description and include visuals if applicable.

### Classification
**MusicNet**:
#### Choice of Model and Algorithms:
**Chosen Model(s)**
- Description of Implementation, hyperparameter selection, computation time, etc.

**GTZAN**:
#### Choice of Model and Algorithms:
**Chosen Model(s)**
- Description of Implementation, hyperparameter selection, computation time, etc.

## Results and Discussion

### MusicNet
- Quantitative metrics
- F1 Scores, confusion matrix, etc.

### GTZAN
- Quantitative metrics
- Accuracy: 90.34%
              precision    recall  f1-score   support

     Class 0       0.90      0.85      0.87       186
     Class 1       0.91      0.97      0.94       199
     Class 2       0.91      0.89      0.90       194
     Class 3       0.92      0.93      0.92       190
     Class 4       0.90      0.83      0.87       189
     Class 5       0.93      0.95      0.94       200
     Class 6       0.89      0.91      0.90       203
     Class 7       0.93      0.90      0.91       244
     Class 8       0.85      0.89      0.87       210
     Class 9       0.90      0.91      0.90       183

    accuracy                           0.90      1998
   macro avg       0.90      0.90      0.90      1998
weighted avg       0.90      0.90      0.90      1998
- F1 Scores, confusion matrix, etc.
-![image](https://github.com/abarton51/CS_4641_Project/assets/73034441/c650ea82-34e3-480f-af20-46ab67a7c203)



### Discussion
**MusicNet**: Data is not distributed well. Need to go actually get more data if we want to reliably do classificaiton on all the composers in the dataset.

**GTZAN**:

**Overall**:

#### Next Steps

# References
[1.]

[2.]

[3.]
