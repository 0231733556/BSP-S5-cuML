from .models import ClassifierWrapper, ClusteringWrapper, train_model,set_defaults,set_log_level, use_accel
from .datagen import generate_classification_data, generate_clustering_data
from sklearn.model_selection import train_test_split
import logging as LOG
SEED = 42

def main () :
    set_log_level(LOG.DEBUG)
    classifierXY = generate_classification_data(n_samples=100_000, n_features=20, n_informative=15, n_classes=2,sparsity=0.0, random_state=42)
    classifierSplit = train_test_split(classifierXY[0], classifierXY[1], test_size=0.2, random_state=42)
    clusteringXY = generate_clustering_data(n_samples=100_000, n_features=50, centers=10, cluster_std=1.0, random_state=42)
    
    setting={
        "classifier": {
            "estimator_name": "random_forest",
            "use_scaler": True,
            "random_state": SEED,
            "n_estimators": 100,
            "max_iter": 1000,
            "probability": True,
        },
        "clustering": {
            "algorithm_name": "kmeans",
            "n_clusters": 3,
            "use_scaler": True,
            "random_state": SEED,
            "algorithm_params": None,
        },
    }
    set_defaults(setting)

    rfc = ClassifierWrapper()
    kmc = ClusteringWrapper()
    resultClassifier=train_model(model=rfc,X=classifierSplit[0],y=classifierSplit[2],X_val=classifierSplit[1],y_val=classifierSplit[3],timing=True)
    resultClustering=train_model(kmc,X=clusteringXY[0],timing=True)
    print("Classifier results:", resultClassifier)
    print("Clustering results:", resultClustering)


if __name__ == "__main__":
    main()