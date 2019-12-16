import xgboost as xgb
from sklearn.model_selection import train_test_split

def get_rating(data):
    '''
    Get the ratings

    :param data: all data
    :return: ratings list
    '''
    rating_list = []
    for i in data:
        if i['reviewTime'] is None or i['gps'] is None:
            continue
        rating_list.append(i['rating'] if i['rating'] is not None else 0)
    return rating_list


def get_feature(data):
    '''

    Get features

    :param data: all data
    :return: feature vectors list
    '''
    feature_list = []
    for i in data:
        if i['reviewTime'] is None or i['gps'] is None:
            continue
        temp = []
        temp.append(0 if i['reviewText'] == None else len(i['reviewText']))
        temp.append(i['gps'][0])
        temp.append(i['gps'][1])
        temp.append(int(i['reviewTime'].split(' ')[-1]))
        feature_list.append(temp)
    return feature_list

def get_mse(ans, label):
    '''

    Calculate the mse

    :param ans: the predictions
    :param label: the true labels
    :return: the mean squared error
    '''
    mse = 0
    for i,j in zip(ans, label):
        mse += (i - j) ** 2
    mse /= len(ans)
    return mse


def grid_search(depth, rate, estimators):
    '''

    Grid search to get best parameters

    :param depth: decision trees' depth
    :param rate: learning rate
    :param estimators: number of estimators
    :return: best parameters model
    '''
    model = xgb.XGBRegressor(max_depth=depth, learning_rate=rate, n_estimators=estimators, silent=False, objective='reg:squarederror')
    model.fit(X_train, y_train)
    ans = model.predict(X_valid)
    return model, ans

if __name__ == '__main__':
    # Read all data from the dataset and split them as train, validation, test
    allReviews = []
    with open('../CA_reviews.txt', encoding='utf-8') as f:
        for l in f:
            allReviews.append(eval(l))

    middleSet, testSet = train_test_split(allReviews, test_size=0.2)
    trainSet, validSet = train_test_split(middleSet, test_size=0.125)

    # Get the train,valid,test sets
    X_train = get_feature(trainSet)
    y_train = get_rating(trainSet)
    X_valid = get_feature(validSet)
    y_valid = get_rating(validSet)
    X_test = get_feature(testSet)
    y_test = get_rating(testSet)

    # Grid search to find best hyper-parameter for the XGBoost model
    best_model = None
    best_mse = 10

    depth = [5,10,15]
    rate = [0.1,0.3,0.6,0.9]
    estimators=[40,80,160]

    for i in depth:
        for j in rate:
            for k in estimators:
                model, ans = grid_search(i,j,k)
                if get_mse(ans, y_valid) < best_mse:
                    best_model = model
                    best_mse = get_mse(ans, y_valid)

    # Using best model for prediction
    pred = best_model.predict(X_test)
    print('Best Mean Squared Error on validation set is: ' + str(best_mse))
    print('Best Mean Squared Error on testing set is: ' + str(get_mse(pred, y_test)))

