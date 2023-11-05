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
import midi_processing
import wav_processing
from midi_roll import roll

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline