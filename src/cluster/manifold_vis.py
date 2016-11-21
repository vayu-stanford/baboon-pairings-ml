import sys
sys.path.append('../common')
import extract
import visualize_labels
from sklearn import manifold
from sklearn.preprocessing import scale

RANDOM_SEED=5

def apply_manifold(X, num_components):
    #X = scale(X)
    m = manifold.TSNE(n_components=num_components, random_state=RANDOM_SEED)
    return m.fit_transform(X)

def manifold_vis_2d(X, y):
    X = apply_manifold(X, 2)
    visualize_labels.plot_2d_labelled(X,y)

def manifold_vis_3d(X, y):
    X = apply_manifold(X, 3)
    visualize_labels.plot_3d_labelled(X,y)

def main():
    include_transformed = True
    (X,y) = extract.generate_labelled_data(include_transformed=include_transformed)
    manifold_vis_2d(X, y)

if __name__=="__main__":
    main()
