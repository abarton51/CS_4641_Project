import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


fig_path = 'src/musicNet/figs/tsne_plots'
data_path = 'src/musicNet/processed_data'
X_train = np.load(data_path + '/train_data_midi.npy')
X_test = np.load(data_path + '/test_data_midi.npy')
y_train = np.load(data_path + '/train_labels_midi.npy')
y_test = np.load(data_path + '/test_labels_midi.npy')

X_full = np.concatenate([X_train, X_test]).astype('int')
y_full = np.concatenate([y_train, y_test]).astype('int')

# 0 -> 0 - Bach
# 1 -> 1 - Beethoven
# 2 -> 2 - Brahms
# 7 -> 3 - Mozart
# 9 -> 4 - Schubert

y_full[y_full==7] = 3
y_full[y_full==9] = 4

tsne = TSNE(n_components=3)
X_embedded = tsne.fit_transform(X_full)
colors = ['dodgerblue', 'blueviolet', 'firebrick', 'darkslategray', 'forestgreen']

def tsne_plot(embd_data, labels, colors=None):
    classes = np.unique(labels)
    dim = embd_data.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in classes:
        embd_data_i = embd_data[labels==i]
        ax.scatter(embd_data_i[:,0], embd_data_i[:,1], embd_data_i[:,2], color=colors[i])
    plt.savefig(fig_path + '/tsne_plot_' + str(dim) + 'd')

tsne_plot(X_embedded, y_full, colors)