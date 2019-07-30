import numpy as np
from scipy.spatial.distance import cdist


def predictAction(testMoment, trainMoments, trainLabels):
    var = np.var(trainMoments, axis=0)
    minDist = np.inf
    minDistInd = -1
    for i in range(len(trainMoments)):
        vec = trainMoments[i]
        vec = np.reshape(vec, (1, 7))
        dist = cdist(np.reshape(testMoment, (1, 7)), vec, 'seuclidean', V=var)
        if dist < minDist:
            minDist = dist
            minDistInd = i
    return trainLabels[minDistInd]

