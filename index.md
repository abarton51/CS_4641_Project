---
title: Overview
layout: home
---
# Introduction/Background:
We are going to perform a classification task on two different music audio datasets, MusicNet and GTZAN. MusicNet consists of 330 .wav and .midi files (per file type) corresponding to 330 separate classical piano compositions belonging to 15 different composers. GTZAN consists of 1000 mean feature matrices stored in .csv files and spectrogram images (per data type) corresponding to 10 genres of music. For MusicNet, the task is to identify the composer for a given input of audio data and for GTZAN the task is to classify the genre of music given an input of audio data. Both of these datasets are taken from [Kaggle](https://www.kaggle.com) and work in classification has recently gotten up to ~92% [[4.]](#references). Previous works struggled getting any model above 80% [[1.]](#references), [[2.]](#references). One study introduced an ensemble decision tree method that outperformed full connected neural networks ([2.]). Results these days outperform them but not much work has been done in using tree classifiers in this problem recently. Therefore, we aim to re-examine decision trees' abilities for this task and attempt to improve upon CNNs.

# Proposed Methods
We plan to primarily explore using neural networks such as fully connected, convolutional, and recurrent neural networks. Within the class of neural networks as methods, we are specifically planning on improving upon CNNs and the combined method illustrated in [[1.]](#references). We plan to compare these methods to other classic methods such as support vector classifiers and decision trees. If permitting, we will construct a super learner model and see if we can get improvements on our results that justifies such a costly ensemble learning method.

# Potential Results/Discussions
In our exploration of our data, we expect to be able to effectively reduce the dimension of the .csv datasets while maintaining its ability to separate classes. For our classification task, we expect to see a combination of CNNs and FFNs to perform the best from previous works with marginal improvements from adding in more spectrogram and image type data.

# References
[1.](https://cs229.stanford.edu/proj2021spr/report2/81973885.pdf) Pun, A., &; Nazirkhanova, K. (2021). Music Genre Classification with Mel Spectrograms and CNN.

[2.](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26583519.pdf) Jain, S., Smit, A., &; Yngesjo, T. (2019). Analysis and Classification of Symbolic Western Classical Music by Composer.

[3.](https://ceur-ws.org/Vol-2718/paper04.pdf) Pál, T., & Várkonyi, D.T. (2020). Comparison of Dimensionality Reduction Techniques on Audio Signals. Conference on Theory and Practice of Information Technologies.

[4.](https://www.kaggle.com/code/imsparsh/gtzan-genre-classification-deep-learning-val-92-4) Gupta, S. (2021). GTZAN-Genre Classification-Deep Learning-Val-92.4%.

## Introduction/Background: 
A quick introduction of your topic and mostly literature review of what has been done in this area. You can briefly explain your dataset and its features here too.

**Problem definition:** Why there is a problem here or what is the motivation of the project?

**Methods:** What algorithms or methods are you going to use to solve the problems. (Note: Methods may change when you start implementing them which is fine). Students are encouraged to use existing packages and libraries (i.e. scikit-learn) instead of coding the algorithms from scratch.

**Potential results and Discussion** (The results may change while you are working on the project and it is fine; that’s why it is called research). A good way to talk about potential results is to discuss about what type of quantitative metrics your team plan to use for the project (i.e. ML Metrics).

At least **three references** (preferably peer reviewed). You need to properly cite the references on your proposal. This part does NOT count towards word limit.

Add proposed **timeline** from start to finish and list each project members’ responsibilities. Fall and Spring semester sample Gantt Chart. This part does NOT count towards word limit.

A **contribution table** with all group members’ names that explicitly provides the contribution of each member in preparing the project task. This part does NOT count towards word limit.

A **checkpoint** to make sure you are working on a proper machine learning related project. You are required to have your dataset ready when you submit your proposal. You can change dataset later. However, you are required to provide some reasonings why you need to change the dataset (i.e. dataset is not large enough because it does not provide us a good accuracy comparing to other dataset; we provided accuracy comparison between these two datasets). The reasonings can be added as a section to your future project reports such as midterm report.

Your group needs to **submit a presentation of your proposal**. Please provide us a public link which includes a **3 minutes recorded video**. I found that OBS Studio and GT subscribed Kaltura are good tools to record your screen. Please make your visuals are clearly visible in your **video presentation**.

**3 MINUTE** is a hard stop. We will NOT accept submissions which are 3 minutes and one second or above. Conveying the message easily while being concise is not easy and it is a great soft skill for any stage of your life, especially your work life.
