import os
import numpy as np
from music21 import converter, stream, note, chord
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# converting midi files to input data
def midi_to_input_data(file_path, num_timesteps=64):
    stream = converter.parse(file_path)
    parts = stream.parts

    # getting the nodes and chords and other music stuff
    elements_to_parse = []
    for part in parts:
        elements_to_parse.extend(part.flat.notesAndRests)

    # list to hold the string representation of elements
    notes = []
    for element in elements_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
        elif isinstance(element, note.Rest):
            notes.append('Rest')

    # making sequeneces
    input_sequences = []
    for i in range(0, len(notes) - num_timesteps, 1):
        sequence = notes[i:i + num_timesteps]
        input_sequences.append(sequence)

    return input_sequences

# creating cnn model
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# preprocessing data and training model
def train_model(dataset_path, num_timesteps=64, epochs=10):
    composers = os.listdir(dataset_path)
    num_composers = len(composers)
    composer_to_index = {composer: idx for idx, composer in enumerate(composers)}

    # loading midi data
    X, y = [], []
    for composer in composers:
        composer_path = os.path.join(dataset_path, composer)
        for file in os.listdir(composer_path):
            file_path = os.path.join(composer_path, file)
            sequences = midi_to_input_data(file_path, num_timesteps)
            X.extend(sequences)
            y.extend([composer_to_index[composer]] * len(sequences))

    # sequences --> numerical format
    unique_elements = list(set(element for sequence in X for element in sequence))
    element_to_index = {element: idx for idx, element in enumerate(unique_elements)}
    X_numerical = np.array([[element_to_index[element] for element in sequence] for sequence in X])

    # labels --> categorical format
    y_categorical = to_categorical(y, num_classes=num_composers)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_numerical, y_categorical, test_size=0.2, random_state=42)

    # reshaping data for cnn input
    input_shape = (num_timesteps, 1, 1)  # Add dimensions for CNN input
    X_train = X_train.reshape(X_train.shape[0], *input_shape)
    X_test = X_test.reshape(X_test.shape[0], *input_shape)

    # creat/compile model
    model = create_model(input_shape, num_composers)

    # train model
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    return model

# replace 'dataset_path' with the path to MIDI dataset
#dataset_path = 'C:\Users\Karpagam\Documents\CLASSES\Fall_2023\CS_4641\final_project' # replace line -- some dummy path for testing data
#dataset_path = 'src\\musicNet\\processed_data\\train_data_midi.npy'
#trained_model = train_model(dataset_path, num_timesteps=64, epochs=10)

# save the trained data?
#trained_model.save('composer_classification_model.h5')
