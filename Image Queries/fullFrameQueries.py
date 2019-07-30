import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from selectRegion import roipoly
from dist2 import dist2
from sklearn.cluster import MiniBatchKMeans
import glob
import pickle as pkl
from scipy import misc
from skimage.color import rgb2gray
import matplotlib.cm as cm
import operator


def buildHist(mat, clf):
    X = mat['descriptors']
    n, _ = np.shape(X)
    if not n:
        return np.zeros(1)
    y = clf.predict(X)
    hist = plt.hist(y, bins=1500)
    return hist[0]

def histSim(hist1, hist2):
    return np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))


clf = pkl.load(open('vocabModel', 'rb'))
framesdir = 'frames/'
siftdir = 'sift/'

fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]


# generate hist for img replace index with query
fname = siftdir + fnames[0]
origMat = scipy.io.loadmat(fname)

hist = buildHist(origMat, clf)

frames = []
# build more hists and compare
count = 1
for file in fnames:
    print 'reading frame %d of %d' %(count, len(fnames))
    count = count + 1
    fname = siftdir + file
    if count != 99:
        mat = scipy.io.loadmat(fname)
    else:
        continue
    compHist = buildHist(mat, clf)
    if not compHist.any():
        print 'empty'
        continue

    score = histSim(hist, compHist)
    print score
    if score > .5:
        frames.append((mat['imname'], score))

frames.sort(key=operator.itemgetter(1))

for i in range(7):
    im = frames[i][0]
    im = framesdir + str(im[0])
    print im
    img = plt.imread(im)
    plt.imshow(img)
    plt.show()
