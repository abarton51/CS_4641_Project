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

# Choose the two composers to compare or comment out and set X = X_full, y = y_full to include all of them

X1 = X_full[np.where(y_full==3)]
X2 = X_full[np.where(y_full==4)]
y1 = y_full[np.where(y_full==3)]
y2 = y_full[np.where(y_full==4)]

X = np.concatenate([X1, X2])
y = np.concatenate([y1, y2])

p = np.random.permutation(X.shape[0])
X = X[p]
y = y[p]

dim=2
tsne = TSNE(n_components=dim)
X_embedded = tsne.fit_transform(X)
colors = ['dodgerblue', 'blueviolet', 'firebrick', 'darkslategray', 'forestgreen']

def tsne_plot(embd_data, labels, name=None, colors=None, dim=2):
    classes = np.unique(labels)
    c = len(colors)
    dim = embd_data.shape[1]
    fig = plt.figure()
    if dim==3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    for i in classes:
        embd_data_i = embd_data[labels==i]
        if dim==3:
            ax.scatter(embd_data_i[:,0], embd_data_i[:,1], embd_data_i[:,2], color=colors[i%c])
        else:
            ax.scatter(embd_data_i[:,0], embd_data_i[:,1], color=colors[i%c])
    if name==None:
        plt.savefig(fig_path + '/tsne_plot_' + str(dim) + 'd')
    else:
        plt.savefig(fig_path + '/tsne_plot_' + name + '_' + str(dim) + 'd')

name = 'Mozart_vs_Schubert'
tsne_plot(X_embedded, y, name=name, colors=colors, dim=dim)