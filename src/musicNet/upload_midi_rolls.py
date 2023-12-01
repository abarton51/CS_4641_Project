import sys
sys.path.append('../')
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from midi_roll import roll

import librosa
import librosa.display
import IPython

import music21

seed = 12
np.random.seed(seed)

data_path = 'C:\\Users\\Teddy\\Documents\\Academics\\Machine Learning\\Projects\\CS_4641_Project\\src\\musicNet\\data'
musicnet_path = data_path + '/musicnet'
path_audio_files = musicnet_path + '/train_data'
midi_path = data_path + '/musicnet_midis'

hop_length = 512
n_fft = 2048

df = pd.read_csv(data_path + '/musicnet_metadata.csv')
composers = np.unique(df['composer'].values)

fail_count = {}
for composer in composers:
    fail_count[composer] = 0

for composer in composers:
    midi_composer_path = midi_path + '/' + composer
    for i, midi_file in enumerate(os.listdir(midi_composer_path)):
        try:
            mid_roll = roll.MidiFile(midi_composer_path + '/' + midi_file)
            K = 16
            # draw piano roll by pyplot
            mid_roll.draw_roll(K, save_roll=data_path + '/midi_rolls/' + composer + '/' + os.path.splitext(midi_file)[0])
        except:
            fail_count[composer] += 1
            continue
print(fail_count)
"""
data_roots = os.listdir(data_path + '/musicnet')
for data_root in data_roots:
    fig = plt.figure()
    ax = fig.add_subplot()
    # Reading one random audio file per composer
    data,sampling_rate = librosa.load(path_audio_files + ".wav")
    
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sampling_rate, hop_length=hop_length)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(mel_spec_db, sr = sampling_rate, hop_length = hop_length, x_axis = 'time', y_axis = 'log',cmap = 'cool',ax=axes[k][j])
    fig.colorbar(img)
    
    ax.set_title(composers)
"""
