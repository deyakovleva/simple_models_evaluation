from sklearn.model_selection import train_test_split


def preprocess(data, features, rubrics):

    rubrics = rubrics.reset_index()
    rubric_dict = {}
    for index, row in rubrics.iterrows():
        rubric_dict[row['rubric_id']] = row['rubric_name']

    features = features.reset_index()
    features_dict = {}
    for index, row in features.iterrows():
        features_dict[row['feature_id']] = row['feature_name']

    # Remove restaurants with unknown 'average_bill' or 'average_bill'>2500
    labels = data[data.columns[-1]].values
    feature_matrix = data[data .columns[:-1]].values
    clean_data = data.dropna(subset=['average_bill'])
    clean_data = data[data['average_bill'] <= 2500.0]

    print(
        f'Number of objects in dataframe after cleaning: {clean_data.shape[0]}')

    clean_data['average_bill'] = (clean_data['average_bill']).astype(int)
    mean = clean_data.groupby('city')['average_bill'].mean()
    print(f'Mean: {mean}')
    median = clean_data.groupby('city')['average_bill'].median()
    print(f'Median: {median}')

    clean_data_train, clean_data_test = train_test_split(
        clean_data, stratify=clean_data['average_bill'], test_size=0.33, random_state=42)
    return clean_data_train, clean_data_test
