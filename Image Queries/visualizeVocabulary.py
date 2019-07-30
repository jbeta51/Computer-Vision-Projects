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



framesdir = 'frames/'
siftdir = 'sift/'

fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

# ********************* RUN K MEANS (comment out when saved) ***************************
#
#
# # build training for kmeans
# fnames = fnames[:1]
# descriptors = list()
# count = 1
# for file in fnames:
#     print 'reading frame %d of %d' %(count, len(fnames))
#     count = count + 1
#     fname = siftdir + file
#     if count != 99:
#         mat = scipy.io.loadmat(fname)
#     else:
#         continue
#
#     for descriptor in mat['descriptors']:
#         descriptors.append(descriptor)

# descriptors = np.array(descriptors)

# ********************* RUN K MEANS (comment out when saved) ***************************
#
# kmeans = MiniBatchKMeans(n_clusters=1500, verbose=2).fit(descriptors)
#
# pkl.dump(kmeans, open('vocabModel', 'wb'))
#
# print 'Thank God'
# exit(0)

# ************************* end of vocab building **************************************

words = {}
clf = pkl.load(open('vocabModel', 'rb'))

# np.random.shuffle(fnames)
word0 = clf.cluster_centers_[1000]
words = []  # ****************************
sample = fnames[500:1000]
count = 1
for file in sample:
    count = count + 1
    fname = siftdir + file
    if count != 99:
        mat = scipy.io.loadmat(fname)
    else:
        continue

    for i in range(len(mat['descriptors'])):
        descriptor = mat['descriptors'][i]
        position = mat['positions'][i]
        scale = mat['scales'][i]
        orient = mat['orients'][i]
        im = mat['imname']
        im = framesdir + str(im[0])
        # thisWord = clf.predict([descriptor])
        distance = dist2(word0.reshape((-1, 1)), descriptor.reshape((-1, 1))).mean()  # ***********************
        words.append(((position, scale, orient, im), distance))

        # if thisWord[0] in words:
        #     words[thisWord[0]] = words[thisWord[0]] + [(position, scale, orient, im)]
        # else:
        #     words[thisWord[0]] = [(position, scale, orient, im)]

words.sort(key=operator.itemgetter(1))

wordPatches = []
for patch, _ in words:
    im = misc.imread(patch[3])
    wordPatches.append(getPatchFromSIFTParameters(patch[0], patch[1], patch[2], rgb2gray(im)))
    if len(wordPatches) > 25:
        break

fig = plt.figure(figsize=(8,8))
cols = 4
rows = 5

for j in range(1, cols*rows + 1):
    img = wordPatches[j-1]
    fig.add_subplot(rows, cols, j)
    plt.imshow(img, cmap=cm.Greys_r)
plt.show()

# for i in range(50):
#     wordPatches = []
#     print i, words[i]
#     title = 'Word ' + str(i)
#     for patch in words[i]:
#         im = misc.imread(patch[3])
#         wordPatches.append(getPatchFromSIFTParameters(patch[0], patch[1], patch[2], rgb2gray(im)))
#         if len(wordPatches) > 25:
#             break
#
#     fig = plt.figure(figsize=(8,8))
#     cols = 4
#     rows = 5
#     for j in range(1, cols*rows + 1):
#         img = wordPatches[j-1]
#         fig.add_subplot(rows, cols, j)
#         plt.imshow(img, cmap=cm.Greys_r)
#     plt.title(title)
#     plt.show()
