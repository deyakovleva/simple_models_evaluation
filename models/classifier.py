from sklearn.base import ClassifierMixin
import numpy as np


class MostFrequentClassifier(ClassifierMixin):
    # Predicts the most frequent
    def fit(self, X=None, y=None):

        unique, counts = np.unique(y, return_counts=True)
        most_frequent_index = np.argmax(counts)
        self.most_frequent = unique[most_frequent_index]
        return self

    def predict(self, X=None):

        return np.full(shape=X.shape[0], fill_value=self.most_frequent)
