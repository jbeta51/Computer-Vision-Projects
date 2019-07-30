import glob
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy
import numpy as np
from computeMHI import computeMHI
from huMoments import huMoments

basedir = './'

actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']

allMHIs = list()

for action in actions:
    subdirname = basedir + action + '/'
    subdirs = os.listdir(subdirname)

    for seq in subdirs:
        # cycle through all sequences for this action category
        directory = subdirname + seq
        mhi = computeMHI(directory)
        allMHIs.append(mhi)

allMHIs = np.stack(allMHIs, axis=2)
np.save('allMHIs.npy', allMHIs)
print 'saved MHIs'

allHus = list()

allMHIs = np.load('allMHIs.npy')
_, _, numSeqs = np.shape(allMHIs)
for i in range(numSeqs):
    mhi = allMHIs[:, :, i]
    huVec = huMoments(mhi)
    allHus.append(huVec)

allHus = np.stack(allHus)
np.save('huVectors.npy', allHus)
print 'saved Hus'
