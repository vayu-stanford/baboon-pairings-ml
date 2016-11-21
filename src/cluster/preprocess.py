import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

rawdata = np.loadtxt('../../data/rawdata.csv', delimiter=',')
#valid_labels=('AF','AS','EU')
#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(68))
#attrs=scale(attrs)
#labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[69],dtype='str')#Gets continents
#for label in np.unique(labels):
    #print label,np.sum(labels==label)
#valid_indices=[i for i in range(labels.shape[0]) if labels[i] in valid_labels]
#labels=labels[valid_indices]
#attrs=attrs[valid_indices,:]

#X = attrs
#y = labels
