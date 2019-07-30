import glob
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy
import numpy as np
from computeMHI import computeMHI
from huMoments import huMoments
from scipy.spatial.distance import cdist


def showNearestMHIs(targetVector, allVectors):
    var = np.var(allVectors, axis=0)
    distances = list()
    for i in range(len(allVectors)):
        vec = allVectors[i]
        vec = np.reshape(vec, (1, 7))
        dist = cdist(np.reshape(targetVector, (1, 7)), vec, 'seuclidean', V=var)
        distances.append((dist, i))
    distances.sort(key=lambda x: x[0])
    return distances[0:5]


hus = np.load('huVectors.npy')
# i = 7
# i = 2
i = 6

testVec = hus[i]

results = showNearestMHIs(testVec, hus)

fig = plt.figure()
columns = 1
rows = 5

mhis = np.load('allMHIs.npy')
for i in range(len(results)):
    mhi = results[i]
    ind = mhi[1]
    fig.add_subplot(rows, columns, i + 1)
    plt.imshow(mhis[:,:,ind])
plt.show()
