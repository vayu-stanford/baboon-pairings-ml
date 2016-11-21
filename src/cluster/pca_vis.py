import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition

valid_labels=('AF','AS','EU')
attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(68))
attrs=scale(attrs)
labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[69],dtype='str')#Gets continents
for label in np.unique(labels):
    print label,np.sum(labels==label)
valid_indices=[i for i in range(labels.shape[0]) if labels[i] in valid_labels]
labels=labels[valid_indices]
attrs=attrs[valid_indices,:]

X = attrs
y = labels

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

colormap = {'AF':'r','EU':'b','AS':'g'}

colors=[]
for yi in y:
    colors.append(colormap[yi])

for label in labels:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean(),
              X[y == label, 2].mean(), label,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
#y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap=plt.cm.spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()
