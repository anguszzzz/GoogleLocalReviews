from sklearn.model_selection import train_test_split
from collections import defaultdict


# This method is used to get a baseline model
if __name__ == '__main__':
    allReviews = []
    reviewCount = 0
    with open('../CA_reviews.txt',encoding='utf-8') as f:
        for l in f:
            allReviews.append(eval(l))

    middleSet,testSet = train_test_split(allReviews,test_size=0.2)
    trainSet,validSet = train_test_split(middleSet,test_size=0.125)

    allRatings = []
    userRatings = defaultdict(list)

    for l in trainSet:
        rating, user, place = l['rating'], l['gPlusUserId'], l['gPlusPlaceId']
        allRatings.append(rating)
        userRatings[user].append(rating)

    globalAverage = sum(allRatings) / len(allRatings)
    userAverage = defaultdict(int)
    for u in userRatings:
        userAverage[u] = sum(userRatings[u]) / len(userRatings[u])


    valid_MSE = 0
    for l in validSet:
        currRating, currUser = l['rating'], l['gPlusUserId']
        if currUser in userAverage:
            valid_MSE += (currRating - userAverage[currUser]) ** 2
        else:
            valid_MSE += (currRating - globalAverage) ** 2

    valid_MSE /= len(validSet)

    print("Mean Square Error on validation set is : ", valid_MSE)

    MSE = 0
    for l in testSet:
        currRating, currUser = l['rating'], l['gPlusUserId']
        if currUser in userAverage:
            MSE += (currRating - userAverage[currUser]) ** 2
        else:
            MSE += (currRating - globalAverage) ** 2

    MSE /= len(testSet)

    print("Mean Square Error on testing set is : ", MSE)

