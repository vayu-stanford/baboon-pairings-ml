import numpy as np
import extract

def whiten(attrs):
    pass

def scale(attrs, scale=True):
    pass

def remove_similar_points(attrs, labels, similarity_measure = 0.01):
    pass

def augment(attrs, labels, augment_label = '1', attr_idxs = None, mult=2, noise_mean=1.0, noise_stdev=0.001):
    if mult < 1:
        raise('Mult must be >= 1')

    if(attr_idxs == None):
        attr_idxs = range(attrs.shape[1])

    new_attrs = np.empty([(mult-1)*attrs.shape[0],attrs.shape[1]])
    new_labels = np.empty([(mult-1)*labels.shape[0]])
    new_labels.fill(augment_label);
    new_arr_idx = 0
    for (idx, label) in enumerate(labels):
        if label == augment_label:
            for m in range(mult):
                for (col_idx, col) in enumerate(attrs[idx]):
                    noise = noise_mean
                    if col_idx in attr_idxs:
                        noise = np.random.normal(noise_mean,.001);
                    new_attrs[new_arr_idx][col_idx] = noise*attrs[idx,col_idx];
                new_arr_idx = new_arr_idx + 1
    attrs = np.concatenate((attrs,new_attrs),axis=0)
    labels = np.concatenate((labels,new_labels),axis=0)
    return(attrs,labels)

def main():
    include_transformed = True
    (attrs,labels) = extract.generate_labelled_data(include_transformed=include_transformed)
    print(attrs.shape)
    (attrs, labels) = augment(attrs,labels, mult=3)

if __name__=="__main__":
    main()
