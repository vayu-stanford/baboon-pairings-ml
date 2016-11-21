import sys
sys.path.append('../common')
import extract
import visualize_labels
from sklearn import decomposition
from sklearn.preprocessing import scale

def apply_pca(X, num_components):
    X = scale(X)
    pca = decomposition.PCA(n_components=num_components)
    pca.fit(X)
    X = pca.transform(X)
    return X

def pca_vis_2d(label_type, include_transformed):
    (X,y) = extract.generate_labelled_data(label_type=label_type,include_transformed=include_transformed)
    X = apply_pca(X, 2)
    visualize_labels.plot_2d_labelled(X,y)

def pca_vis_3d(label_type, include_transformed):
    (X,y) = extract.generate_labelled_data(label_type=label_type, include_transformed=include_transformed)
    X = apply_pca(X, 3)
    visualize_labels.plot_3d_labelled(X,y)

pca_vis_2d('consort', True)
pca_vis_2d('consort', False)
