from sklearn.metrics import mean_squared_error, balanced_accuracy_score
import numpy as np
import pandas as pd
from data_preprocessing import preprocess
from models.regressor import MeanRegressor
from models.classifier import MostFrequentClassifier
from models.city_mean_regressor import CityMeanRegressor
import argparse


def parser_args(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=[
        'mean_regressor', 'most_frequent_classifier', 'city_mean_regressor'], help="Choose a model")

    global FLAGS
    FLAGS = parser.parse_args(argv)


class Model():

    def __init__(self):

        dataset_path = 'dataset/'
        self.data = pd.read_csv(dataset_path + 'organisations.csv')
        self.features = pd.read_csv(dataset_path + 'features.csv')
        self.rubrics = pd.read_csv(dataset_path + 'rubrics.csv')

    def eval(self, FLAGS):

        clean_data_train, clean_data_test = preprocess(
            self.data, self.features, self.rubrics)

        if FLAGS.model == 'mean_regressor':
            reg = MeanRegressor()
            reg.fit(y=clean_data_train['average_bill'])
            y_pred_reg = reg.predict(clean_data_test['average_bill'])
            RMSE_reg = np.sqrt(mean_squared_error(
                clean_data_test['average_bill'], y_pred_reg))
            print(f'RMSE_reg: {RMSE_reg}')

        if FLAGS.model == 'most_frequent_classifier':
            clf = MostFrequentClassifier()
            clf.fit(y=clean_data_train['average_bill'])
            y_pred_clf = clf.predict(clean_data_test['average_bill'])
            RMSE_clf = np.sqrt(mean_squared_error(
                clean_data_test['average_bill'], y_pred_clf))
            BAS_clf = balanced_accuracy_score(
                clean_data_test['average_bill'], y_pred_clf)
            print(f'RMSE_clf: {RMSE_clf}, BAS_clf: {BAS_clf}')

        if FLAGS.model == 'city_mean_regressor':
            cmr = CityMeanRegressor()
            cmr.fit(X=clean_data_train['city'], y=clean_data_train)
            y_pred_cmr = cmr.predict(clean_data_test)
            RMSE_cmr = np.sqrt(mean_squared_error(
                clean_data_test['average_bill'], y_pred_cmr))
            print(f'RMSE_cmr: {RMSE_cmr}')

        if FLAGS.model is None:  # TODO catch error
            print('Error')


def main(FLAGS):

    model = Model()
    model.eval(FLAGS)


if __name__ == '__main__':

    parser_args()
    main(FLAGS)
