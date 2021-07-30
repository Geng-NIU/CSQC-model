from sklearn.feature_selection import RFE
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from time import time

t0 = time()
"""loading saved pickle data: data_X, data_y"""

data_X = preprocessing.minmax_scale(data_X, feature_range=(0, 1))
data_X = preprocessing.scale(data_X)
data_y = np.ravel(data_y, order='C')
min_features_to_select = 1
step = 1
estimator = ExtraTreesClassifier(n_estimators=100, random_state=111)
"""step 1"""
rfecv = RFECV(estimator=estimator, min_features_to_select=min_features_to_select, step=step, cv=StratifiedKFold(3),
              scoring='accuracy')
rfecv.fit(data_X, data_y)
print("Optimal number of features : %d" % rfecv.n_features_)

fig, ax = plt.subplots(figsize=(5, 2.5))
ax.plot(np.linspace(min_features_to_select, len(rfecv.grid_scores_)*step, len(rfecv.grid_scores_)),
        rfecv.grid_scores_, 'k')
ax.set_xlabel('Number of features selected', fontsize=10)
ax.set_ylabel('Model Accuracy', fontsize=10)
ax.grid(True, which='both')
plt.tight_layout()
plt.show()
# fig.save

"""step 2"""
selector = RFE(estimator=estimator, n_features_to_select=20, step=1)
selector = selector.fit(data_X, data_y)
inputs_binary_model = data_X[:, selector.support_]
print(inputs_binary_model.shape)
feature_selected = estimator.fit(inputs_binary_model, data_y)  # extra tree
importance_score = feature_selected.feature_importances_
ranking = np.argsort(-importance_score)
print(ranking)
