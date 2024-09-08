#!/usr/bin/env python


"""
Semi K-Means
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_is_fitted

class SupervisedKMeans(ClassifierMixin, KMeans):

    """Supervised Learning by KMeans

    Center-classifier: x -> argmin_k |x-mu_k|
    
    Attributes:
        centers_ (array): classification center
        cluster_centers_ (array): == centers_
        n_classes (int): the number of classes
    """

    labels_ = None

    def fit(self, X, y):
        if self.labels_ is None:
            self.labels_ = np.unique(y)
        self.centers_ = np.array([np.mean(X[y==c], axis=0) for c in self.labels_])
        self.cluster_centers_ = self.centers_
        self.n_classes = len(self.labels_)
        return self

    def predict(self, X):
        ed = euclidean_distances(X, self.cluster_centers_)
        return np.asarray([self.labels_[k] for k in np.argmin(ed, axis=1)])

    def score(self, X, y):
        y_ = self.predict(X)
        return np.mean(y == y_)


class SemiKMeans(SupervisedKMeans):
    # Semi-Supervised KMeans

    def fit(self, Xl, yl, Xu):
        """To fit the semisupervised model
        
        Args:
            Xl (array): input variables with labels
            yl (array): labels
            Xu (array): input variables without labels
        
        Returns:
            the model
        """
        labels_ = np.unique(yl)
        if not hasattr(self, 'labels') or self.labels_ is None:
            self.labels_ = np.arange(self.n_clusters)
        else:
            assert all(c in self.labels_ for c in labels_), 'yl has an element not in `labels`!'

        X = np.row_stack((Xl, Xu))
        
        n1 = self.n_clusters - len(labels_)
        mu0 = SupervisedKMeans().fit(Xl, yl).centers_
        if n1:
            centers, indices = kmeans_plusplus(Xu, n_clusters=n1)
            self.cluster_centers_ = np.row_stack((centers, mu0))
        else:
            self.cluster_centers_ = mu0

        return self._fit(Xl, yl, Xu, self.cluster_centers_, self.labels_)

    def _fit(self, Xl, yl, Xu, cluster_centers, labels):
        X = np.row_stack((Xl, Xu))

        for _ in range(self.max_iter):
            ED = euclidean_distances(Xu, cluster_centers)
            yu = [labels[k] for k in np.argmin(ED, axis=1)]
            y = np.concatenate((yl, yu))
            cluster_centers = np.array([np.mean(X[y==c], axis=0) for c in labels])

        self.cluster_centers_ = cluster_centers
        return self

    def partial_fit(self, *args, **kwargs):
        check_is_fitted(self, ('cluster_centers_',))
        return self._fit(Xl,yl, Xu, self.cluster_centers_, self.labels_)
