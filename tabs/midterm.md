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
MusicNet is a collection of 330 freely licensed classical music recordings, together with over 1 million annotated labels indicating the precise time of each note in every recording, the instrument that plays each note, and the note's position in the metrical structure of the composition. The labels are acquired from musical scores aligned to recordings by dynamic time warping. The labels are verified by trained musicians; a labeling error rate of 4% has been estimated. The MusicNet labels are offered to the machine learning and music communities as a resource for training models and a common benchmark for comparing results.
https://www.kaggle.com/datasets/imsparsh/musicnet-dataset 

**GTZAN**: Description and link to dataset.
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification 

## Problem Definition
Why there is a problem here? What is the motivation for the project? 

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
We experimented with a feedforward neural network. As the input data consisted only of extracted features that were not spatially or temporally related, a CNN, RNN, transformer, or other kind of model was not needed. We chose the neural network as we believed we had sufficient training samples and that this function may be sufficiently complex for a neural network to approximate. We will explore this further by implementing other non-NN machine learning models in the future.

To begin, the 3-second variant was experimented with. We made a 10% split from our training dataset to create a validation set to detect when overfitting was occurring. Early stopping was implemented such that training would halt as soon as 3 epochs had the validation loss not decrease. We employed a softmax activation function for our final layer (as this is a multiclass single-label classification problem) and the categorical cross-entropy loss. For hidden layers, we used ReLU. For a network this shallow, no vanishing gradients were observed, making the use of a ReLU variant (such as LeakyReLU) unnecessary. The Adam optimizer was used with a learning rate of 10^-3.

The iteration over model architectures began with a single hidden layer of 32 neurons. From there, the number of neurons was increased, and as signs of overfitting were noticed, dropout regularization was added in. Not only did this prevent overfitting, but it also improved model performance compared to less parameter-dense networks, likely a consequence of breaking co-adaptations. Ultimately, performance peaked with a network consisting of 2 hidden layers, each with a 0.5 dropout and 512 neurons each. 

Results using 3-second samples:
- Quantitative metrics
![image](https://github.com/abarton51/CS_4641_Project/assets/73034441/5c6815ca-1d82-4f7c-8eeb-184b48bffcf8)

- F1 Scores, confusion matrix, etc.
Loss:
![image](https://github.com/abarton51/CS_4641_Project/assets/73034441/c650ea82-34e3-480f-af20-46ab67a7c203)
Confusion Matrix:
![image](https://github.com/abarton51/CS_4641_Project/assets/73034441/85ccf7ba-80c5-42f2-9e19-5b554963ec12)

However, it is important to note that performance did not climb significantly from a single-hidden-layer network with just 128 neurons and dropout. After this, additional improvements to model capacity provided diminishing returns for the increased needs for computing.

With similar experimentation, it was found (along with optimizer parameters) that a batch size of 32 resulted in the best performance, reaching 90+% accuracy on the well-balanced test set.
When dealing with the 30-second variant, the number of training samples drastically reduced, making overfitting a concern. While we ultimately ended up using virtually the same setup (only a smaller batch size, this time 16), we had to make changes to the size and number of layers.

It was found that the ceiling for performance was a model that had two 64-neuron hidden layers with a dropout of 0.5 each. Anything more complex, and the model would simply start to overfit (triggering early stopping) before it could properly converge on a good solution.

In any case, the smaller dataset and smaller model resulted in severely degraded test set performance, with the neural network only achieving an accuracy of just over 70%. 

Results using 30-second samples:
- Quantitative metrics
![image](https://github.com/abarton51/CS_4641_Project/assets/73034441/8f992238-62dd-456e-ae58-522a68920771)

- F1 Scores, confusion matrix, etc.
Loss:
![image](https://github.com/abarton51/CS_4641_Project/assets/73034441/95134070-2529-4ef7-9728-8a18fae2e1ca)
Confusion Matrix:
![image](https://github.com/abarton51/CS_4641_Project/assets/73034441/4a25e36f-0db0-4aed-80a3-1dd9d65f00c9)



### Discussion
**MusicNet**: Data is not distributed well. Need to go actually get more data if we want to reliably do classificaiton on all the composers in the dataset.

**GTZAN**: 
When perfecting the accuracy of our models we came across a few notable mistakes. 
- Often times rock music would be misclassified as disco or metal.
- A large amount of jazz samples were misclassified as classical.

  
**Overall**:

#### Next Steps

# References
[1.]

[2.]

[3.]
