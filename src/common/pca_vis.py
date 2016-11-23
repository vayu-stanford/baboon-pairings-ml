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

def pca_vis_2d(X, y):
    X = apply_pca(X, 2)
    visualize_labels.plot_2d_labelled(X,y)

def pca_vis_3d(X, y):
    X = apply_pca(X, 3)
    visualize_labels.plot_3d_labelled(X,y)

def main():
    include_transformed=True
    (X,y) = extract.generate_labelled_data(include_transformed=include_transformed)
    pca_vis_2d(X,y)

if __name__=="__main__":
    main()
