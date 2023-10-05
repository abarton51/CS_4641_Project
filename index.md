---
title: Proposal
layout: home
---
# Introduction/Background:
We are going to perform a classification task on two different music audio datasets, MusicNet and GTZAN. MusicNet consists of 330 WAV and MIDI files (per file type) corresponding to 330 separate classical piano compositions belonging to 15 different composers. GTZAN consists of 1000 mean feature matrices stored in CSV files and spectrogram PNG images (per file type) corresponding to 10 genres of music. For MusicNet, the task is to identify the composer for a given input of audio data and for GTZAN the task is to classify the genre of music given an input of audio data. Both of these datasets are taken from [Kaggle](https://www.kaggle.com) and work in classification has recently gotten up to ~92% [[4.]](#references). Previous works struggled getting any model above 80% [[1.]](#references), [[2.]](#references). One study introduced a gradient boosted ensemble decision tree method called LightGBM that outperformed fully connected neural networks [[2.]](#references). Results these days outperform them but not much recent work has been done in using tree classifiers in this problem and most implementations appear to focus on neural network implementations. Therefore, we aim to re-examine decision trees' abilities for this task and attempt to improve upon neural network results. Additionally, in our exploratory data analysis and data pre-processing we would like to consider non-linear, as well as linear, dimensionality reduction techniques. We would like to evaluate these different methods similar to Pal et al., [[3]](#references), by reducing dimensions to a specified number, running a clustering algorithm on the data, and then evaluating results posthoc. In their results, t-SNE consistently outperformed other dimension reduction techniques. Therefore, we plan to use t-SNE in order to better understand and visualize our data in addition to principle components analysis (PCA).

# Proposed Methods
We plan to primarily explore using neural networks such as fully connected and convolutional. Within the class of neural networks as methods, we are specifically planning on improving upon CNNs and the combined method illustrated in [[1.]](#references). We will also explore and evaluate gradient boosted decision trees such as XGBoost and LightGBM. We plan to compare these methods to other classic methods such as support vector classifiers and logistic regression. If permitting, we may construct a super learner model and see if we can get improvements on our results that justifies such a costly ensemble learning method.

# Potential Results/Discussions
In our exploration of our data, we hope to be able to effectively reduce the dimension of the CSV datasets while maintaining some ability to separate data belonging to different classes. For each of our classification tasks, we expect to see a combination of CNNs and MLPs to perform the best from previous works with marginal improvements from adding in more spectrogram and image type data. Despite neural networks being expected to outperform other methods, we believe that gradient boosted decision trees may perform similarly and may provide benefits in lower training time and being able to analyze decision splits in parent nodes.

# Checkpoints

(TODO: THIS IS JUST A ROUGH TEMPLATE)
1. Dataset completion
2. Midterm report
3. Final report
(etc.)

# Contribution Table

| Contributor Name | Contribution Type  |
|-------------------|-------------------|
| Austin Barton        | Github, Proposal, Dataset Choices|
| Aditya Radhakrishnan | GanttChart |
| Isabelle Murray      | GanttChart, Contribution Table |
| Karpagam Karthikeyan | Video script, Video presentation |
| Niki (Keyang) Lu     | Video presentation |

![Excel Chart](https://github.com/abarton51/CS_4641_Project/raw/main/assets/images/GanttChart_ClassicalficationMusic-1.png)

![Excel Chart](https://github.com/abarton51/CS_4641_Project/raw/main/assets/images/GanttChart_ClassicalficationMusic-2.png)

![Excel Chart](https://github.com/abarton51/CS_4641_Project/raw/main/assets/images/GanttChart_ClassicalficationMusic-3.png)


# References
[1.](https://cs229.stanford.edu/proj2021spr/report2/81973885.pdf) Pun, A., &; Nazirkhanova, K. (2021). Music Genre Classification with Mel Spectrograms and CNN.

[2.](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26583519.pdf) Jain, S., Smit, A., &; Yngesjo, T. (2019). Analysis and Classification of Symbolic Western Classical Music by Composer.

[3.](https://ceur-ws.org/Vol-2718/paper04.pdf) Pál, T., & Várkonyi, D.T. (2020). Comparison of Dimensionality Reduction Techniques on Audio Signals. Conference on Theory and Practice of Information Technologies.

[4.](https://www.kaggle.com/code/imsparsh/gtzan-genre-classification-deep-learning-val-92-4) Gupta, S. (2021). GTZAN-Genre Classification-Deep Learning-Val-92.4%.
