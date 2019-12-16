from collections import defaultdict
from sklearn.model_selection import train_test_split

def calMSE(dataset, alpha, beta_u, beta_b):
    '''

    Calculate the mean squared error

    :param dataset: the train,valid or test set
    :param alpha: the alpha
    :param beta_u: the beta_u
    :param beta_b: the beta_b
    :return: the mse
    '''
    avg_beta_u = sum(beta_u[k] for k in beta_u)/len(beta_u)
    avg_beta_p = sum(beta_b[k] for k in beta_b) / len(beta_b)
    
    MSE = 0
    for l in dataset:
        user,place,rating = l['gPlusUserId'],l['gPlusPlaceId'],l['rating']
        
        predRating = alpha             + (beta_u[user] if user in beta_u else avg_beta_u)             + (beta_b[place] if place in beta_b else avg_beta_p)
        MSE += (predRating - rating) ** 2
    MSE /= len(dataset)
    return MSE

def gridSearch(lams):
    '''

    Grid search for best parameters

    :param lams: tuple of lambdas
    :return: best model
    '''
    lam1, lam2 = lams
    prev_alpha = 0
    prev_beta_u = defaultdict(int)
    prev_beta_p = defaultdict(int)
    prev_MSE = 0

    while (True):
        alpha = 0
        beta_u = defaultdict(int)
        beta_p = defaultdict(int)
        for l in trainSet:
            user, place, rating = l['gPlusUserId'], l['gPlusPlaceId'], l['rating']
            alpha += rating - (prev_beta_u[user] + prev_beta_p[place])
        alpha /= len(allRatings)

        for user in user_places:
            for place in user_places[user]:
                beta_u[user] += allRatings[user + place] - (alpha + prev_beta_p[place])
            beta_u[user] /= (lam1 + len(user_places[user]))

        for place in place_users:
            for user in place_users[place]:
                beta_p[place] += allRatings[user + place] - (alpha + prev_beta_u[user])
            beta_p[place] /= (lam2 + len(place_users[place]))

        MSE = calMSE(trainSet, alpha, beta_u, beta_p)

        if abs(prev_MSE - MSE) < 0.0001:
            print('lambda is:', lams)
            print("Alpha is " , alpha)
            print("Training Mean Square Error is ", MSE)
            validMSE = calMSE(validSet, alpha, beta_u, beta_p)
            print("Validation Mean Square Error is ", validMSE)
            return (validMSE, alpha, beta_u, beta_p)
        prev_MSE = MSE
        prev_alpha = alpha
        prev_beta_u = beta_u
        prev_beta_p = beta_p

if __name__ == '__main__':
    # Read all data from the dataset and split them as train, validation, test
    allReviews = []
    with open('../CA_reviews.txt', encoding='utf-8') as f:
        for l in f:
            allReviews.append(eval(l))

    middleSet, testSet = train_test_split(allReviews, test_size=0.2)
    trainSet, validSet = train_test_split(middleSet, test_size=0.125)


    allRatings = defaultdict(int)
    user_places = defaultdict(set)
    place_users = defaultdict(set)
    for l in trainSet:
        user, place, rating = l['gPlusUserId'], l['gPlusPlaceId'], l['rating']
        user_places[user].add(place)
        place_users[place].add(user)
        allRatings[user + place] = rating


    lams = ((l1 / 10.0,l2 / 10.0) for l1 in range(25,35) for l2 in range(25,35))
    lamPair = min(lams, key=gridSearch)
    curr_mse, alpha, beta_u, beta_i = gridSearch(lamPair)
    print("Best lambda is: ",lamPair)
    print("Using this lambda, the mean square error on the validation set is: " ,curr_mse)

    testMSE = calMSE(testSet, alpha, beta_u, beta_i)
    print("Test Mean Square Error is ", testMSE)

