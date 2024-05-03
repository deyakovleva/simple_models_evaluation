import pandas as pd
from sklearn.base import RegressorMixin


class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):

        y = pd.DataFrame(y, columns=['average_bill'])
        new_data = pd.concat([X, y], axis=1)
        self.city_means = new_data.groupby('city')['average_bill'].mean()
        return self

    def predict(self, X=None):

        return X['city'].map(self.city_means)
