import glob
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy
import numpy as np


def moment(img, i, j):
    rows, cols = np.shape(img)
    sum = 0
    for r in range(rows):
        for c in range(cols):
            sum += r**i * c**j * img[r,c]
    return sum

def mu(img, p, q):
    xBar = moment(img, 1, 0) / moment(img, 0, 0)
    yBar = moment(img, 0, 1) / moment(img, 0, 0)

    sum = 0
    rows, cols = np.shape(img)
    for r in range(rows):
        for c in range(cols):
            sum += (r - xBar)**p * (c - yBar)**q * img[r,c]
    return sum


def huMoments(mhi):
    mu02 = mu(mhi, 0, 2)
    mu03 = mu(mhi, 0, 3)
    mu11 = mu(mhi, 1, 1)
    mu12 = mu(mhi, 1, 2)
    mu20 = mu(mhi, 2, 0)
    mu21 = mu(mhi, 2, 1)
    mu30 = mu(mhi, 3, 0)

    h1 = mu20 + mu02
    h2 = (mu20 - mu02)**2 + (4 * mu11)**2
    h3 = (mu30 - 3 * mu12)**2 + (3 * mu21 - mu03)**2
    h4 = (mu30 + mu12)**2 + (mu21 + mu03)**3
    h5 = (mu30 - 3 * mu12) * (mu30 + mu12) * ((mu30 + mu12)**2 - 3 * (mu21 + mu03)**2) + \
         (3 * mu21 - mu12) * (mu21 + mu03) * (3 * (mu30 + mu12) ** 2 - (mu21 + mu03) ** 2)
    h6 = (mu20 - mu02) * ((mu30 + mu12)**2 - (mu21 + mu03)**2) + \
         4 * mu11 * (mu30 + mu12) * (mu21 + mu03)
    h7 = (3 * mu21 - mu03) * (mu30 + mu12) * ((mu30 + mu12)**2 - 3 * (mu21 + mu03)**2) - \
         (mu30 - 3 * mu12) * (mu21 + mu03) * (3 * (mu30 + mu12)**2 - (mu21 + mu03)**2)
    h = [h1, h2, h3, h4, h5, h6, h7]
    return np.array(h)



