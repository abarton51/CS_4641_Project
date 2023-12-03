import sys
sys.path.append('../')
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import librosa
import librosa.display
import IPython

import musicnet4641

seed = 12
np.random.seed(seed)

data_path = 'C:\\Users\\Teddy\\Documents\\Academics\\Machine Learning\\Projects\\CS_4641_Project\\src\\musicNet\\data'
melspec_path = data_path + '/mel_specs'
musicnet_path = data_path + '/musicnet'
train_data_path = musicnet_path + '/train_data'
test_data_path = musicnet_path + '/test_data'

hop_length = 512
n_fft = 2048

df = pd.read_csv(data_path + '/musicnet_metadata.csv')
composers = np.unique(df['composer'].values)
id_dict = musicnet4641.create_id_dict(df)

fail_count = {}
for composer in composers:
    fail_count[composer] = 0
total_fail_count = 0

def upload_melspecs(data_path, save_path, id_dict, fail_count):
    for i, wav_file in enumerate(os.listdir(data_path)):
        data, sampling_rate = librosa.load(data_path + '/' + wav_file)
        mel_spec = librosa.feature.melspectrogram(y=data, sr=sampling_rate, hop_length=hop_length)
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        
        
        img = librosa.display.specshow(mel_spec_db, sr=sampling_rate, hop_length=hop_length, x_axis='time', y_axis='log', cmap='cool')
        plt.colorbar(img)
        
        composer = None
        for key in id_dict.keys():
            if int(os.path.splitext(wav_file)[0]) in id_dict[key]:
                composer = str(key)
                break
            
        plt.savefig(save_path + '/' + composer + '/' + os.path.splitext(wav_file)[0])
        plt.close()
    print(fail_count)
    
upload_melspecs(data_path=train_data_path, save_path=melspec_path, id_dict=id_dict, fail_count=total_fail_count)

"""
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    upload_melspecs(data_path=train_data_path, save_path=melspec_path, id_dict=id_dict, fail_count=total_fail_count)
"""