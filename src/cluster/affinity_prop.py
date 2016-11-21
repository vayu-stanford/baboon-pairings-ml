import sys
sys.path.append('../common')
import extract
import pca_vis
import manifold_vis
from sklearn.cluster import AffinityPropagation

RANDOM_SEED=5

def apply_affinity_prop_consort(include_transformed):
    (X,y) = extract.generate_labelled_data(valid_labels=['1'], label_type='consort',include_transformed=include_transformed)
    am = AffinityPropagation()
    preds = am.fit_predict(X)
    return (X, preds)

def main():
    include_transformed=True
    (X, preds) = apply_affinity_prop_consort(include_transformed=include_transformed)
    # PCA is faster, but manifold gives better separation
    # pca_vis.pca_vis_2d(X, preds)
    manifold_vis.manifold_vis_3d(X, preds)

if __name__=="__main__":
    main()
