import sys
sys.path.append('../common')
import extract
import pca_vis
import manifold_vis
from sklearn.cluster import KMeans

RANDOM_SEED=5

def apply_kmeans_consort(num_clusters, include_transformed):
    (X,y) = extract.generate_labelled_data(valid_labels=['1'], label_type='consort',include_transformed=include_transformed)
    km = KMeans(n_clusters=num_clusters, random_state=RANDOM_SEED)
    preds = km.fit_predict(X)
    return (X, preds, km.cluster_centers_)

def apply_kmeans_nonconsort(num_clusters, include_transformed):
    (X,y) = extract.generate_labelled_data(valid_labels=['0'], label_type='consort',include_transformed=include_transformed)
    print(extract.get_feature_names(valid_labels=['0'], label_type='consort',include_transformed=include_transformed))
    km = KMeans(n_clusters=num_clusters, random_state=RANDOM_SEED)
    preds = km.fit_predict(X)
    return (X, preds, km.cluster_centers_)

def main():
    include_transformed=True
    (X, preds, cluster_centers) = apply_kmeans_consort(5,include_transformed=include_transformed)
    # PCA is faster, but manifold gives better separation
    # pca_vis.pca_vis_2d(X, preds)
    print(cluster_centers)
    manifold_vis.manifold_vis_3d(X, preds)

if __name__=="__main__":
    main()
