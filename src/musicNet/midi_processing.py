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

def open_midi(midi_path):
    """
    One line method to parse through and read MIDIs

    Args:
        midi_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    mf = music21.midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()      

    return music21.midi.translate.midiFileToStream(mf)

def list_instruments(midi):
    partStream = midi.parts.stream()
    print("List of instruments found on MIDI file:")
    for p in partStream:
        aux = p
        print (p.partName)

def extract_notes(midi_part):
    parent_element = []
    ret = []
    for nt in midi_part.flat.notes:        
        if isinstance(nt, music21.note.Note):
            ret.append(max(0.0, nt.pitch.ps))
            parent_element.append(nt)
        elif isinstance(nt, music21.chord.Chord):
            for pitch in nt.pitches:
                ret.append(max(0.0, pitch.ps))
                parent_element.append(nt)
    
    return ret, parent_element

def print_parts_countour(midi):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    minPitch = music21.pitch.Pitch('C10').ps
    maxPitch = 0
    xMax = 0
    # Drawing notes.
    for i in range(len(midi.parts)):
        top = midi.parts[i].flat.notes                  
        y, parent_element = extract_notes(top)
        if (len(y) < 1): continue
            
        x = [n.offset for n in parent_element]
        ax.scatter(x, y, alpha=0.6, s=7)
        
        aux = min(y)
        if (aux < minPitch): minPitch = aux
            
        aux = max(y)
        if (aux > maxPitch): maxPitch = aux
            
        aux = max(x)
        if (aux > xMax): xMax = aux
    
    for i in range(1, 10):
        linePitch = music21.pitch.Pitch('C{0}'.format(i)).ps
        if (linePitch > minPitch and linePitch < maxPitch):
            ax.add_line(mlines.Line2D([0, xMax], [linePitch, linePitch], color='red', alpha=0.1))            

    plt.ylabel("Note index (each octave has 12 notes)")
    plt.xlabel("Number of quarter notes (beats)")
    plt.title('Voices motion approximation, each color is a different instrument, red lines show each octave')
    plt.show()