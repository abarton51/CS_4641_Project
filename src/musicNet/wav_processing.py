#!C:/Users/Teddy/anaconda3/envs/cs4641_env
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import os
from tqdm import tqdm

import librosa
import librosa.display
from mido import MidiFile
from midi_roll import roll
import music21
#import midi

import IPython

def extract_classes(path):
    """
    Analyze the contents of a directory containing WAV or PNG files and generate a dictionary of genres and their counts.

    Args:
        path (str): The path to the directory containing WAV or PNG files to be analyzed.

    Returns:
        dict: A dictionary where keys represent 'genre' and values represent the number of files in the directory
              associated with each 'genre'.
    """
    classes = {}
    i = 0
    for piece in os.listdir(path):
        classes[piece] = i
        i += 1
    return classes

def extract_wav(path, classes, wav_duration, duration=2, sr=1000):
    """
    Extract audio data and labels from a directory of WAV files.

    Args:
        path (str): The path to the directory containing WAV files to be processed.
        classes (list): List of the classes.
        wav_duration (int): The duration (in seconds) of each training example.
        duration (int, optional): The desired duration (in seconds) for audio segments. Defaults to 2 seconds.
        sr (int, optional): The sample rate of the WAV files. Defaults to 1000 samples per second.

    Returns:
        X (numpy.ndarray): A 2D numpy array containing audio data, where each row represents an audio segment.
        Y (numpy.ndarray): A 1D numpy array containing labels for the audio segments.
    """
    dim = sr * duration
    samples_per_wav = wav_duration // duration
    X = np.zeros((1, dim))
    Y = np.zeros((1))
    for piece in tqdm(os.listdir(path)):
        piece_path = os.path.join(path, piece)
        file_path = os.path.join(path, piece)
        try:
            x, _ = librosa.load(file_path, sr=sr)
            x = np.pad(x, (0, 30), mode='constant', constant_values=0)
            sample_x = x[0: samples_per_wav * dim].reshape(samples_per_wav, dim)
            sample_y = np.ones((samples_per_wav)) * classes[piece]
            X = np.concatenate((X, sample_x))
            Y = np.concatenate((Y, sample_y))
        except:
            print("librosa load failed on " + file_path)
    return X[1:,:], Y[1:]

def play_buttons(path, pieces, composers=None):
    for i, id_i in enumerate(pieces):
        print(composers[i])
        # Reading one random audio file per composer
        data,sampling_rate = librosa.load(path + '/' + str(id_i) + '.wav')
        display(IPython.display.Audio(data, rate = sampling_rate))