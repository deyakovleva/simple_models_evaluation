import numpy as np
from scipy.stats import mode
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor


class MeanRegressor(RegressorMixin):
    # Predicts the mean
    def fit(self, X=None, y=None):

        self.mean_ = y.mean()
        return self

    def predict(self, X=None):

        return np.array(X.shape[0]*[self.mean_])
