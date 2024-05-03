from sklearn.metrics import mean_squared_error, balanced_accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.regressor import MeanRegressor
from models.classifier import MostFrequentClassifier
from models.city_mean_regressor import CityMeanRegressor

base = '/root/workspace/yandex_ml_course/intro/'
data = pd.read_csv(base + 'organisations.csv')
features = pd.read_csv(base + 'features.csv')
rubrics = pd.read_csv(base + 'rubrics.csv')

rubrics = rubrics.reset_index()

rubric_dict = {}
for index, row in rubrics.iterrows():
    rubric_dict[row['rubric_id']] = row['rubric_name']

features = features.reset_index()
features_dict = {}
for index, row in features.iterrows():

    features_dict[row['feature_id']] = row['feature_name']


# Уберите из них все заведения, у которых средний чек неизвестен или превышает 2500.
# Пока есть опасение, что их слишком мало, чтобы мы смогли обучить на них что-нибудь.
labels = data[data.columns[-1]].values
feature_matrix = data[data .columns[:-1]].values
clean_data = data.dropna(subset=['average_bill'])
clean_data = data[data['average_bill'] <= 2500.0]

print(f'Number of objects in dataframe after cleaning: {clean_data.shape[0]}')


clean_data['average_bill'] = (clean_data['average_bill']).astype(int)
mean = clean_data.groupby('city')['average_bill'].mean()
print(f'Mean: {mean}')
median = clean_data.groupby('city')['average_bill'].median()
print(f'Median: {median}')

clean_data_train, clean_data_test = train_test_split(
    clean_data, stratify=clean_data['average_bill'], test_size=0.33, random_state=42)

reg = MeanRegressor()
reg.fit(y=clean_data_train['average_bill'])

clf = MostFrequentClassifier()
clf.fit(y=clean_data_train['average_bill'])

y_pred_reg = reg.predict(clean_data_test['average_bill'])
y_pred_clf = clf.predict(clean_data_test['average_bill'])


RMSE_reg = np.sqrt(mean_squared_error(
    clean_data_test['average_bill'], y_pred_reg))

RMSE_clf = np.sqrt(mean_squared_error(
    clean_data_test['average_bill'], y_pred_clf))

BAS_clf = balanced_accuracy_score(clean_data_test['average_bill'], y_pred_clf)


cmr = CityMeanRegressor()
cmr.fit(X=clean_data_train['city'], y=clean_data_train)

y_pred_cmr = cmr.predict(clean_data_test)
RMSE_cmr = np.sqrt(mean_squared_error(
    clean_data_test['average_bill'], y_pred_cmr))
print(
    f'RMSE_reg: {RMSE_reg}, RMSE_clf: {RMSE_clf}, RMSE_cmr: {RMSE_cmr}, BAS_clf: {BAS_clf}')
