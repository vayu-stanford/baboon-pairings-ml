import sys
sys.path.append('../common')
import extract
import visualize_labels
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

RANDOM_SEED=5

def apply_kmeans_nonconsort(num_clusters, include_transformed):
    (X,y) = extract.generate_labelled_data(valid_labels=['0'], label_type='consort',include_transformed=include_transformed)

    km = KMeans(n_clusters=num_clusters, random_state=RANDOM_SEED)
    preds = km.fit_predict(X)
    X = scale(X)
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)
    visualize_labels.plot_3d_labelled(X,preds)

apply_kmeans_nonconsort(3,include_transformed=True)
