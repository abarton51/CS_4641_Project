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
Why there is a problem here? What is the motivation of the project? 

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
- Quantitative metrics
- 3-second samples:
<img src="/assets/images/gtzan-accuracy-3sec.JPG" alt="drawing" width="200"/>


- F1 Scores, confusion matrix, etc.
Confusion Matrix:
<img src="/assets/images/gtzan_mlp_3secs_confmatrix.png" alt="drawing" width="200"/>

Loss:
<img src="/assets/images/gtzan_mlp_3secs_loss.png" alt="drawing" width="200"/>

- 30-second samples:
<img src="/assets/images/gtzan-accuracy-30sec.JPG" alt="drawing" width="200"/>

- F1 Scores, confusion matrix, etc.
Confusion Matrix:
<img src="/assets/images/gtzan-30sec-confmatrix.png" alt="drawing" width="200"/>

Loss: 
<img src="/assets/images/gtzan_mlp_30secs_loss.png" alt="drawing" width="200"/>


### Discussion
**MusicNet**: Data is not distributed well. Need to go actually get more data if we want to reliably do classificaiton on all the composers in the dataset.

**GTZAN**: 
While perfecting the accuracy of our model, we came across a few notable mistakes:
-Rock music would often be misclassified as disco or metal. 
-A large number of jazz music samples were misclassified as classical.

**Overall**:

#### Next Steps

# References
[1.]

[2.]

[3.]
