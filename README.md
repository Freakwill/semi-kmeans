# semi-kmeans
Semi supervised k-means algo.


```python
#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from semi_kmeans import *
from sklearn.cluster import KMeans

from sklearn import datasets
digits = datasets.load_digits()
X, y = digits.data, digits.target

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X)

n_clusters = 4
X = X[y<n_clusters]; y = y[y<n_clusters]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_train, y_train, test_size=0.95)

# create semi k-means
skm = SemiKMeans(n_clusters=n_clusters)
skm.fit(X_labeled, y_labeled, X_unlabeled)

km = KMeans(n_clusters=n_clusters)
km.fit(X)
clf = SupervisedKMeans()
clf.fit(X_labeled, y_labeled)

# print(f"""
# # clusters: 10
# # samples: {X_labeled.shape[0]} + {X_unlabeled.shape[0]}
# SemiKMeans: {km.score(X_test, y_test)}
# SupervisedKMeans: {skm.score(X_test, y_test)}
# """)

from utils import visualize

import matplotlib.pyplot as plt
plt.style.use('b_style.mplstyle')

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)

x1lim = X[:, 0].min(), X[:, 0].max()
x2lim = X[:, 1].min(), X[:, 1].max()

ax.scatter(*X_unlabeled.T, color='grey', alpha=0.4)
visualize(ax, skm, X_labeled, y_labeled, x1lim=x1lim, x2lim=x2lim, N1=400, N2=400, boundary=True, boundary_kw={'s':1.6, 'alpha': 0.8, 'c':'red'})
visualize(ax, clf, X_labeled, y_labeled, x1lim=x1lim, x2lim=x2lim, N1=50, N2=40, scatter=False, boundary_kw={'s':1.6, 'alpha': 0.5, 'c':'blue'})
visualize(ax, km, X, y, N1=60, N2=50, scatter=False, boundary=False, background=True, background_kw={'alpha': 0.05, 'marker': 's'})
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.plot([0],[0], 'r-', label=f'semi-K-means cluster（test：{skm.score(X_test, y_test):.4}）', alpha=0.7)
ax.plot([0],[0], 'b:', label=f'K-means classifer（test：{clf.score(X_test, y_test):.4}）', alpha=0.4)
ax.legend()
plt.savefig('../src/semi-kmeans.png')
plt.show()

```
