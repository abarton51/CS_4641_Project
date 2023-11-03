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

def create_id_dict(df):
    """
    Create a dictionary with composer names as keys and their songs as values.

    Args:
        df (pandas.DataFrame): DataFrame that contains the composer names and id values.
        
    Returns:
        dict (dictionary): A dictionary where composer names are keys, and their associated songs are values.
    """
    composers = np.unique(df['composer']).astype(str)
    id_dict = dict()
    for i in range(len(composers)):
        id_dict[composers[i]] = df.loc[df['composer'] == composers[0], 'id']
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