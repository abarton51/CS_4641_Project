#!C:/Users/Teddy/anaconda3/envs/cs4641_env
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import os
from tqdm import tqdm

import wav_processing

import librosa
import librosa.display
from mido import MidiFile
from midi_roll import roll
import music21
#import midi

import IPython

def create_id_dict(df):
    """
    Create a dictionary with composer names as keys and their songs as values.

    Args:
        df (pandas.DataFrame): DataFrame that contains the composer names and id values.
        
    Returns:
        dict (dictionary): A dictionary where composer names are keys, and their associated songs are values.
    """
    composers = np.unique(df['composer'].values).astype(str)
    id_dict = dict()
    for i in range(len(composers)):
        id_dict[composers[i]] = df.loc[df['composer'].values == composers[i]]['id'].values
    return id_dict

def rand_id_sample(dict):
    """
    Return random sample of id values for each composer.
    
    Args:
        dict (dictionary): A dictionary where composer names are keys, and their associated songs are values.
    
    Returns:
        random_id_sample (list): A list of the id values for randomly sample songs for each composer.
    """
    random_id_sample = []
    composers = list(dict.keys())
    for i in range(len(composers)):
        curr_values_list = list(dict[composers[i]])
        c = len(curr_values_list)
        random_id_sample.append(curr_values_list[np.random.randint(0, c, size=1)[0]])
    return random_id_sample

def load_audio_data(df, path, wav_duration=30, duration=2, sr=10, train=True, split_samples=False):
    """
    Load WAV data.

    Args:
        df (Pandas.DataFrame): DataFrame with information on composer and song id number.
        path (str): path into the folder that contains the train or test folder with the data.
        wav_duration (int): The duration (in seconds) of each training example.
        duration (int, optional): The desired duration (in seconds) for audio segments. Defaults to 2 seconds.
        sr (int, optional): The sample rate of the WAV files. Defaults to 1000 samples per second. Deafults to 10.
        train (bool, optional): Specify to load train or test data. Defaults to True.
        split_samples (bool, optional): Specify to split each song into samples or have one sample per song. Defaults to False.

    Returns:
        X: Data sample matrix
        y: Label vector for each sample
    """
    dim = sr * duration
    samples_per_wav = wav_duration // duration
    
    if split_samples:
        X = np.zeros((1, dim))
        y = np.zeros((1))
    else:
        X, y = [], []
    
    data_dict = create_id_dict(df)
    class_dict = {}
    cl_idx = 0
    for composer in list(data_dict.keys()):
        class_dict[composer] = cl_idx
        cl_idx+=1

    if train:
        file_path = path + '/train_data'
    else:
        file_path = path + '/test_data'
    
    for composer in tqdm(list(data_dict.keys())):
        for song in data_dict[composer]:
            try:
                x, _ = librosa.load(file_path + '/' + str(song) + '.wav', sr=2)
                x = np.pad(x, (0, 100), mode='constant', constant_values=0)
                if split_samples:
                    sample_x = x[0: samples_per_wav * dim].reshape(samples_per_wav, dim)
                    sample_y = np.ones((samples_per_wav)) * class_dict[composer]
                    X = np.concatenate((X, sample_x))
                    y = np.concatenate((y, sample_y))
                else:
                    sample_x = x[0: dim]
                    sample_y = class_dict[composer]
                    X.append(sample_x)
                    y.append(sample_y)
            except:
                print("librosa load failed on " + file_path + '/' + str(song) + '.wav')
                
    if split_samples:
        return X[1:,:], y[1:]
    
    return np.array(X), np.array(y)