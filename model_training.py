import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from time import time
from sklearn.metrics import roc_auc_score, roc_curve, auc
""" this module is for training the ML model"""


t0 = time()
# =====================================================
""" loading input data: data_X, data_y """

# normalization
data_X = preprocessing.minmax_scale(data_X, feature_range=(0, 1))
data_X = preprocessing.scale(data_X)
# ==================================================================================================
# split dataset
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=3)
X_val_1, X_test_1, y_val_1, y_test_1 = train_test_split(X_test, y_test, test_size=0.5, random_state=3)
# ========================================================================================

# =======================
""" k-neighbor """
# =======================
knn = KNeighborsClassifier(n_neighbors=3,
                           leaf_size=30,
                           algorithm='auto').fit(X_train, y_train)
print('knn:', accuracy_score(knn.predict(X_test), y_test))
tn, fp, fn, tp = confusion_matrix(y_test, knn.predict(X_test)).ravel()
print("tn: %i, fp: %i, fn: %i, tp: %i" % (tn, fp, fn, tp))
true_positive = tp
ppv = tp / (tp + fp)
npv = tn / (fn + tn)
accuracy = (tp + tn) / (tp + fp + fn + tn)
gm = np.sqrt(tp * tn)
print("test_1 KNN ppv: %f, npv: %f, accuracy: %f, gm: %f" % (ppv, npv, accuracy, gm))


# ====================================
""" MLP: multi-layer perception """
# ====================================
mlp = MLPClassifier(solver='adam', alpha=1e-5,
                    learning_rate_init=0.1,
                    hidden_layer_sizes=(5, 5),
                    random_state=1).fit(X_train, y_train)
print('mlp accuracy:', mlp.score(X_test, y_test))
tn, fp, fn, tp = confusion_matrix(y_test, mlp.predict(X_test)).ravel()
print("tn: %i, fp: %i, fn: %i, tp: %i" % (tn, fp, fn, tp))
ppv = tp / (tp + fp)
npv = tn / (fn + tn)
accuracy = (tp + tn)/(tp + fp + fn + tn)
gm = np.sqrt(tp * tn)
print(" ppv: %f, npv: %f, accuracy: %f, gm: %f" % (ppv, npv, accuracy, gm))


"""================= unsupervised =========================="""


# defining transfer function: -1-->1 & 1-->0
def transfer_1(a):
    for i in range(len(a)):
        if a[i] == -1:
            a[i] = 1
        elif a[i] == 1:
            a[i] = 0
    return a


# =================================
""" Isolation forest """
# =================================
IsF = IsolationForest(max_samples=10,
                      n_estimators=1000,
                      max_features=X_train.shape[1],
                      random_state=222,
                      contamination='auto').fit(X_train)

tn, fp, fn, tp = confusion_matrix(y_test, transfer_1(IsF.predict(X_test))).ravel()
print("tn: %d; fp: %d; fn: %d; tp: %d" % (tn, fp, fn, tp))
ppv = tp/(tp + fp)
npv = tn/(fn + tn)
accuracy = (tp + tn)/(tp + fp + fn + tn)
gm = np.sqrt(tp*tn)
print("test IsolationForest ppv: %f, npv: %f, accuracy: %f, gm: %f" % (ppv, npv, accuracy, gm))


# =================================
""" K-Means """
# =================================
kmeans = KMeans(
                n_clusters=2,
                init='random',
                max_iter=1000,
                algorithm='auto',
                copy_x=True,
                n_init=50,
                random_state=0
                ).fit(X_train)

tn, fp, fn, tp = confusion_matrix(y_test, kmeans.predict(X_test)).ravel()
print("tn: %d; fp: %d; fn: %d; tp: %d" % (tn, fp, fn, tp))
ppv = tp/(tp + fp)
npv = tn/(fn + tn)
accuracy = (tp + tn)/(tp + fp + fn + tn)
gm = np.sqrt(tp*tn)
print("test_1 K-Means ppv: %f, npv: %f, accuracy: %f, gm: %f" % (ppv, npv, accuracy, gm))


duration = time() - t0
print("duration: %f" % duration)


