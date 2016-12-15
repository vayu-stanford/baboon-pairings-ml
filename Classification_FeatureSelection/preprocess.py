import numpy as np
#import extract
from sklearn import decomposition
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, squareform

# This WILL modify features beyond recognition
def whiten(attrs):
    attrs = scale(attrs)
    pca = decomposition.PCA(n_components=attrs.shape[1], whiten = True)
    pca.fit(attrs)
    attrs = pca.transform(attrs)
    return attrs

def stdize(attrs, attr_idxs = None):
    return scale(attrs) # just standardize for now

def remove_similar_points(attrs, labels, ids, similarity_measure = 0.01):
    dist_vec = pdist(attrs,'euclidean')
    dist_mtx = squareform(dist_vec)
    removal_set = set()
    for (idx_1, label ) in enumerate(labels):
        for idx_2 in range(idx_1):
            if(label != labels[idx_2] and dist_mtx[idx_1, idx_2] < similarity_measure):
                removal_set.add(idx_1)
                removal_set.add(idx_2)
    attrs = np.delete(attrs,list(removal_set),0)
    labels = np.delete(labels,list(removal_set),0)
    if ids != None:
        ids = np.delete(ids,list(removal_set),0)
    return(attrs,labels,ids)

def augment(attrs, labels, augment_label = 1, attr_idxs = None, mult=2, noise_mean=1.0, noise_stdev=0.001):
    if mult < 1:
        raise('Mult must be >= 1')

    if(attr_idxs == None):
        attr_idxs = range(attrs.shape[1])
    pos_count = np.where(labels==augment_label)[0].shape[0]
    new_attrs = np.empty([(mult-1)*pos_count,attrs.shape[1]])
    new_labels = np.empty([(mult-1)*pos_count])
    new_labels.fill(augment_label);
    new_arr_idx = 0
    for (idx, label) in enumerate(labels):
        if label == augment_label:
            for m in range(mult-1):
                for (col_idx, col) in enumerate(attrs[idx]):
                    noise = noise_mean
                    if col_idx in attr_idxs:
                        noise = np.random.normal(noise_mean,.001);
                    new_attrs[new_arr_idx][col_idx] = noise*attrs[idx,col_idx];
                
                new_arr_idx = new_arr_idx + 1
    attrs = np.concatenate((attrs,new_attrs),axis=0)
    labels = np.concatenate((labels,new_labels),axis=0)

    return(attrs,labels)

def preprocess_data(attrs, labels, ids=None, augment_attr_idxs = None, whitening=False):
    if(augment_attr_idxs == None):
        augment_attr_idxs=range(1,attrs.shape[1]) # don't add noise to conceptive
    attrs = stdize(attrs);
    (attrs, labels,ids) = remove_similar_points(attrs,labels, ids, similarity_measure=0.03)
    if(whitening):
        attrs = whiten(attrs);
    '''(attrs, labels) = augment(attrs,labels, attr_idxs=augment_attr_idxs,  mult=2)'''
    return (attrs,labels,ids)

def main():
    include_transformed = True
    (attrs,labels) = extract.generate_labelled_data(include_transformed=include_transformed)
    preprocess_data(attrs, labels, None)

if __name__=="__main__":
    main()
