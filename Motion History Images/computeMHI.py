import glob
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy
import numpy as np

def computeMHI(dir):
    depthfiles = glob.glob(dir + '/' + '*.pgm')
    depthfiles = numpy.sort(depthfiles)
    mhi = np.zeros(np.shape(imread(depthfiles[0])))
    for t in range(len(depthfiles)):
        depth = imread(depthfiles[t])
        forground = depth < 40000
        background = depth >= 40000
        depth[forground] = 1
        depth[background] = 0

        mhi[forground] = t
        mhi[background] = mhi[background]
    mhi = mhi / np.max(mhi)
    return mhi
