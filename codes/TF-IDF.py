from sklearn.model_selection import train_test_split
import numpy
from collections import defaultdict
import string
from nltk.stem.porter import *
import xgboost as xgb

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

def feature(datum, coeff):
    '''

    Get features for modelling

    :param datum: data sample
    :param coeff: frequency coefficient
    :return: feature vector
    '''
    feat = [0] * len(words)
    r = ''.join([c for c in datum['reviewText'].lower() if not c in punctuation])
    words_temp = r.split()
    wordNum = len(words_temp)
    for i in range(len(words_temp) - 1):
        w = words_temp[i] + " " + words_temp[i + 1]
        if w in wordSet:
            idf = numpy.log10(docNum / wordDoc[w])
            tf = wordFrequency[coeff][w] / wordNum
            feat[wordId[w]] = tf * idf
    feat.append(1)  # offset
    return feat

def get_tf_feat(data):
    '''

    Translate total data into feature vectors

    :param data: all dataset
    :return: the feature vectors
    '''
    X = []
    y = []
    docId = 0
    for d in data:
        if d['reviewText'] is None:
            continue
        X.append(feature(d, docId))
        docId += 1
        y.append(d['rating'])
    return X, y

def grid_search(depth, rate, estimators):
    '''

    Grid search to get best parameters

    :param depth: decision trees' depth
    :param rate: learning rate
    :param estimators: number of estimators
    :return: best parameters model
    '''
    temp_mse = best_mse
    model = xgb.XGBRegressor(max_depth=depth, learning_rate=rate, n_estimators=estimators, silent=False, objective='reg:squarederror')
    model.fit(X_tf_train, y_tf_train)
    ans = model.predict(X_tf_valid)
    return model, ans


if __name__ == '__main__':

    # Read all data from the dataset and split them as train, validation, test
    allReviews = []
    with open('../CA_reviews.txt', encoding='utf-8') as f:
        for l in f:
            allReviews.append(eval(l))

    middleSet, testSet = train_test_split(allReviews, test_size=0.2)
    trainSet, validSet = train_test_split(middleSet, test_size=0.125)


    # Mapping the document no, word frequency and other information useful for getting tf-idf
    docNo = 0
    wordDoc = defaultdict(int)
    wordFrequency = defaultdict(dict)
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()

    for d in allReviews:
        if d['reviewText'] is None:
            continue
        r_sub = ''.join([c for c in d['reviewText'].lower() if not c in punctuation])
        appeared = set()
        temp = r_sub.split()
        for i in range(len(temp)-1):
            w = temp[i] + " " + temp[i+1]
            if not wordFrequency[docNo].get(w):
                wordFrequency[docNo][w] = 0
            wordFrequency[docNo][w] += 1
            if w in appeared:
                continue
            appeared.add(w)
            wordDoc[w] += 1
        docNo += 1

    # Get the most popular 1000 bigrams
    wordCount = defaultdict(int)
    for d in allReviews:
        if d['reviewText'] is None:
            continue
        r = ''.join([c for c in d['reviewText'].lower() if not c in punctuation])
        words = r.split()
        for i in range(len(words)-1):
            wordCount[words[i] + " " + words[i+1]] += 1

    mostPopular = [(wordCount[w] , w) for w in wordCount]
    mostPopular.sort()
    mostPopular.reverse()

    words = [x[1] for x in mostPopular[:1000]]
    # TF-IDF as feature for modelling
    docNum = docNo
    wordId = dict(zip(words, range(len(words))))
    wordSet = set(words)

    # Translate reviews data into feature vectors
    X, y = get_tf_feat(allReviews)

    middleset,X_tf_test,y_middleset,y_tf_test = train_test_split(X,y,test_size=0.2)
    X_tf_train,X_tf_valid,y_tf_train,y_tf_valid = train_test_split(middleset, y_middleset, test_size=0.125)

    # Grid search to find the best model
    best_model = None
    best_mse = 10

    depth = [5,10,15]
    rate = [0.1,0.3,0.6,0.9]
    estimators=[40,80,160]

    for i in depth:
        for j in rate:
            for k in estimators:
                model, ans = grid_search(i,j,k)
                if get_mse(ans, y_tf_valid) < best_mse:
                    best_model = model
                    best_mse = get_mse(ans, y_tf_valid)

    # Report mse for validation and testing sets
    pred = best_model.predict(X_tf_test)
    print('Best Mean Squared Error on validation set is: ' + str(best_mse))
    print('Best Mean Squared Error on testing set is: ' + str(get_mse(pred, y_tf_test)))

