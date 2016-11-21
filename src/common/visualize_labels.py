import matplotlib.pyplot as plt
import matplotlib.cm 
from mpl_toolkits.mplot3d import Axes3D

def plot_2d_labelled(X,y):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.cla()

    colormap = {}
    cm = matplotlib.cm.get_cmap('gist_rainbow')
    label_set = set(y)
    for (i,label) in enumerate(label_set):
        colormap[label]=cm(1.*i/len(label_set))

    print colormap

    colors=[]
    for yi in y:
        colors.append(colormap[yi])

    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()

def plot_3d_labelled(X,y):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    colormap = {}
    cm = matplotlib.cm.get_cmap('gist_rainbow')
    label_set = set(y)
    for (i,label) in enumerate(label_set):
        colormap[label]=cm(1.*i/len(label_set))

    colors=[]
    for yi in y:
        colors.append(colormap[yi])

    for label in label_set:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean(),
                  X[y == label, 2].mean(), label,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap=plt.cm.spectral)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()
