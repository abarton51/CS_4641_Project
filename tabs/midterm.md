---
layout: default
title: 2. Midterm Report
---
# CS 4641 Machine Learning, Team 6, Midterm Report
Members: Austin Barton, Karpagam Karthikeyan, Keyang Lu, Isabelle Murray, Aditya Radhakrishan

## Introduction
Our project's goal is to perform two different classification tasks on two different music audio datasets, MusicNet and GTZAN. GTZAN consists of 1000 mean feature matrices stored in CSV files and spectrogram PNG images (per file type) corresponding to 10 genres of music. For MusicNet, the task is to identify the composer for a given input of audio data and for GTZAN the task is to classify the genre of music given an input of audio data. Both of these datasets are taken from [Kaggle](https://www.kaggle.com) and work in classification has recently gotten up to ~92% [[4.]](#references). Previous works struggled getting any model above 80% [[1.]](#references), [[2.]](#references). One study introduced a gradient boosted ensemble decision tree method called LightGBM that outperformed fully connected neural networks [[2.]](#references). Results these days outperform them but not much recent work has been done in using tree classifiers in this problem and most implementations appear to focus on neural network implementations. Therefore, we aim to re-examine decision trees' abilities for this task and attempt to improve upon neural network results. Additionally, in our exploratory data analysis and data pre-processing we would like to consider non-linear, as well as linear, dimensionality reduction techniques. We would like to evaluate these different methods similar to Pal et al., [[3]](#references), by reducing dimensions to a specified number, running a clustering algorithm on the data, and then evaluating results posthoc. In their results, t-SNE consistently outperformed other dimension reduction techniques. Therefore, we plan to use t-SNE in order to better understand and visualize our data in addition to principle components analysis (PCA).

### Datasets
**MusicNet**: We took this data from [Kaggle](kaggle.com). [MusicNet](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset) is an audio dataset consisting of 330 WAV and MIDI files corresponding to 10 mutually exclusive classes. Each of the 330 WAV and MIDI files (per file type) corresponding to 330 separate classical compositions belong to 10 different composers from the classical and baroque periods. The total size of the dataset is approximately 33 GB and has 992 files in total. 330 of those are WAV, 330 are MIDI, 1 NPZ file of MusicNet features stored in a NumPy array, and a CSV of metadata. For this portion of the project, we essentially ignore the NPZ file and explore our own processing and exploration of the WAV and MIDI data for a more thorough understanding of the data and the task.

**GTZAN**: We took this data from [Kaggle](kaggle.com). [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) is an audio dataset containing a collection of 10 genres with 100 audio files each, all having a length of 30 seconds. It also contains a visual representation of each audio file in the form of a Mel Spectrogram and 2 CSV files that contain features for the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3-second audio files.

## Problem Definition
The problem that we want to solve is the classification of music data into specific categories (composers for MusicNet and genres for GTZAN). Essentially, the existing challenge is to improve previous accuracy benchmarks, especially with methods beyond neural networks, and to explore alternative models like decision trees. Our motivation for this project was to increase classification accuracy, improve the potential of decision trees in this domain, and to better understand and interpret the models that we chose to use. Our project aims to contribute to the field of music classification and expand the range of effective methodologies for similar tasks.

Despite the dominance of neural networks' in recent works, there's motivation to enhance their performance and explore if combining methods can gain better results. hte references and readings suggest that decision trees, especially gradient boosted ones, might perform comparably and offer advantages in terms of training time and interpretability. Based on this, the project aims to effectively reduce the dimensionality of the datasets, enhancing the understanding and visualization of the data using techniques like t-SNE and PCA. 

## Methods

### Exploratory Data Analysis (EDA)
**MusicNet**:
For MusicNet, we thoroughly explore visualization of both WAV and MIDI data files in various ways, using our own created methods, open-source programs, and a mix of the two. We will first go through the MIDI file exploration for Beethoven's 3rd Movement of the famous Moonlight Sonata composition. Something very important to be aware of is how many different MIDI file formats exist and how different some MIDI files may be in certain portions of their structure. The software we used to visualize and parse through MIDI files are open-source and can be found at [python-midi](https://github.com/vishnubob/python-midi/) and [music21](https://web.mit.edu/music21/). In the following example, the MIDI file is broken down into a left hand and right hand portion of the piano solo played. Below is a sample of the right hand and left hand from measures 1 to 10 shown in separate plots, in their respective order.

<img src="/assets/images/beethoven_ms_mvmt3_righthand_measure1to10.png" alt="drawing" width="200"/>
<img src="/assets/images/beethoven_ms_mvmt3_lefthand_measure1to10.png" alt="drawing" width="200"/>

If you've every listened to this song, you'll immediately recognize the overall pattern of the notes. Below is a sample of the left and right hand notes from measure 1 to 6.

<img src="/assets/images/beethoven_ms_mvmt3_partscontour_measures1to6.png" alt="drawing" width="200"/>

Below we show the frequencies of certain pitches and quarter length notes at certain pitches.

<img src="/assets/images/beethoven_ms_mvmt3_frequency_quarterlength_pitch.png" alt="drawing" width="200"/>
<img src="/assets/images/beethoven_ms_mvmt3_pitchclass_frequency.png" alt="drawing" width="200"/>

Below is a an example of the pitch frequency in the vertical axis and the two coordinates along the horizontal directions are pitch and note length.

<img src="/assets/images/beethoven_ms_mvmt3_3dbars.png" alt="drawing" width="200"/>

**GTZAN**:

### Data Preprocessing
**MusicNet**:
For the MIDI files, we aimed to create an algorithm to parse through MIDI files and convert into row vectors to be stored into a data matrix $X$. We utilize the MIDI parsing software from [python-midi](https://github.com/vishnubob/python-midi) to parse through the MIDI files and obtain tensors of float values that corresond to the instrument, the pitch of the note, and the loudness of the note. Each MIDI file generated a (**I**\times **P**\times **A**) tensor stored as a 3-D numpy array where $I$ is the number of instruments, $P$ is the total number of pitches, which range from (1-128), and the total number of quarter notes in the piece **A**. **I** is held as a constant of 16. For any instrument not played, it simply stores a matrix of zeroes. Additionally, the number of quarter notes in each piece is vastly different. Therefore, we require a way to process this data in a way that is homogenous in its dimensions.

We take the average values of each piece across the 3rd dimension (axis = 2) generating a 2-D array of size 16x128 where each entry is the average float value for that corresponindg pitch and instrument across every note in the piece. From here, we flatten the 2-D array to generate a 1-D array of size 16*128 = 2048, where each block of values 128 entries long corresponds to an instrument. This flattening procedure respects the order of the instruments in a consistent manner across all MIDI files and is capable of storing instrument information for all MIDI files within the domain of the 16 instruments. The 16 instruments consist of the most common instruments in classical music including piano, violin, bass, cello, drums, voice, etc. Although the memory is costly, and the non-zero entries in each row vector are quite sparse, we determined that this procedure would be a viable method to maintain information in a manner that is feasible and reasonably efficient for our task.

In summary, we parse through each MIDI file and undergo a basic algorithm to generate row vectors of float values for each composition. We do this for each MIDI file and generate a data matrix X_{MIDI}\in R^{330x2048} stored as a 2-D array of float values. This data matrix X_{MIDI} is what we will process through supervised models in the future and is the data we further explore with Principal Component Analysis detailed in the next section for MusicNet.

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

## Results and Discusssion

### MusicNet
- Quantitative metrics
- F1 Scores, confusion matrix, etc.

### GTZAN
We experimented with a feedforward neural network. As the input data consisted only of extracted features that were not spatially or temporally related, a CNN, RNN, transformer, or other kind of model was not needed. We chose the neural network as we believed we had sufficient training samples and that this function may be sufficiently complex for a neural network to approximate. We will explore this further by implementing other non-NN machine learning models in the future.

To begin, the 3-second variant was experimented with. We made a 10% split from our training dataset to create a validation set to detect when overfitting was occurring. Early stopping was implemented such that training would halt as soon as 3 epochs had the validation loss not decrease. We employed a softmax activation function for our final layer (as this is a multiclass single-label classification problem) and the categorical cross-entropy loss. For hidden layers, we used ReLU. For a network this shallow, no vanishing gradients were observed, making the use of a ReLU variant (such as LeakyReLU) unnecessary. The Adam optimizer was used with a learning rate of 10^-3.

The iteration over model architectures began with a single hidden layer of 32 neurons. From there, the number of neurons was increased, and as signs of overfitting were noticed, dropout regularization was added in. Not only did this prevent overfitting, but it also improved model performance compared to less parameter-dense network, likely a consequence of breaking co-adaptations. Ultimately, performance peaked with a network consisting of 2 hidden layers, each with a 0.5 dropout and 512 neurons each. 

Quantitative metrics
- 3-second samples:
- <img src="/assets/images/gtzan-accuracy-3sec.JPG" alt="drawing" width="200"/>

F1 Scores, confusion matrix, etc.
- Confusion Matrix:
<img src="/assets/images/gtzan_mlp_3secs_confmatrix.png" alt="drawing" width="200"/>

- Loss:
<img src="/assets/images/gtzan_mlp_3secs_loss.png" alt="drawing" width="200"/>

However, it is important to note that performance did not climb significantly from a single-hidden-layer network with just 128 neurons and dropout. After this, additional improvements to model capacity provided diminishing returns for the increased needs for computing.

With similar experimentation, it was found (along with optimizer parameters) that a batch size of 32 resulted in the best performance, reaching 90+% accuracy on the well-balanced test set.
When dealing with the 30-second variant, the number of training samples drastically reduced, making overfitting a concern. While we ultimately ended up using virtually the same setup (only a smaller batch size, this time 16), we had to make changes to the size and number of layers.

It was found that the ceiling for performance was a model that had two 64-neuron hidden layers with a dropout of 0.5 each. Anything more complex, and the model would simply start to overfit (triggering early stopping) before it could properly converge on a good solution.

In any case, the smaller dataset and smaller model resulted in severely degraded test set performance, with the neural network only achieving an accuracy of just over 70%. 

Quantitative metrics
- 30-second samples:
- <img src="/assets/images/gtzan-accuracy-30sec.JPG" alt="drawing" width="200"/>

F1 Scores, confusion matrix, etc.
- Confusion Matrix:
- <img src="/assets/images/gtzan-30sec-confmatrix.png" alt="drawing" width="200"/>

Loss: 
- <img src="/assets/images/gtzan_mlp_30secs_loss.png" alt="drawing" width="200"/>


### Discussion
**MusicNet**: After performing much of the data visualization on the data that we collected from the Kaggle dataset, we noticed that the data is not distributed well. We found that the reason for this distribution is because we need to get more data if we want to reliably do classification on all the composers in the dataset. For instance, we noticed that certain composers like Bach had over 100 songs that we were able to use as data points whereas other composers like Haydn had less than 5 songs within the given dataset, which likely caused this disparity in the results. Because of this, we plan on adding more songs from the different composers to account for this disparity our results. We also found that PCA results of the MIDI data are promising and show separability in the data. In addition to using more sophisticated models, we plan on using a multi-modal classification model that processes MIDI numerical data and also possibly including images of MIDI roles and/or mel-spectrogram graphs. This entails processing not only the raw numerical MIDI data but also considering supplementary information in the form of images, such as MIDI rolls or mel-spectrogram graphs. 

**GTZAN**: One reoccurring error involved the misclassification of rock music samples. Our model had a tendency to mistakenly label rock compositions as either disco or metal, where this misclassification indicates a potential overlap in the feature space or characteristic patterns shared between rock, disco, and metal genres. The challenge lies in distinguishing subtle distinctions that define each genre, such as rhythm patterns, instrumentations, or tonal characteristics, which might be shared or confused by our model. Addressing this misclassification will require a more nuanced feature representation and a deeper understanding of the distinctive elements that differentiate rock from disco and metal. 

Another notable misclassification involved a significant number of jazz music samples being incorrectly labeled as classical. This misclassification hints at shared features or structural elements between jazz and classical music that pose challenges for our model. Improving the model's ability to discern the subtle nuances that distinguish jazz from classical music will be crucial for rectifying this specific misclassification. 

**Overall**: Based on both the MusicNet and GTZAN datasets, we noticed that the uneven distribution of data among the composers in MusicNet has suggested the recognition of potential biases, leading us to plan to address this disparity by increasing the dataset with more compositions from various composers. Additionally, we plan to incorporate numerical MIDI data as well as images like mel-spectrogram graphs. For GTZAN, we propose solutions that involve refining our understanding of distinctive elements when discerning musical nuances, especially in rock and jazz. Based on these results, we can improve our model accuracy and robustness in completing music classification tasks.

## Next Steps

# References
[1.] Pun, A., &; Nazirkhanova, K. (2021). Music Genre Classification with Mel Spectrograms and CNN

[2.] Jain, S., Smit, A., &; Yngesjo, T. (2019). Analysis and Classification of Symbolic Western Classical Music by Composer.

[3.] Pál, T., & Várkonyi, D.T. (2020). Comparison of Dimensionality Reduction Techniques on Audio Signals. Conference on Theory and Practice of Information Technologies.

