import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from dist2 import dist2

mat = scipy.io.loadmat('twoFrameData.mat')
im1 = mat['im1']
im2 = mat['im2']

plt.imshow(im1)
MyROI = roipoly(roicolor='r')
ind = MyROI.getIdx(im1, mat['positions1'])

matches = []
for descriptor in range(len(mat['descriptors2'])):
    for i in ind:
        vector1 = np.array(mat['descriptors1'][i]).reshape((-1, 1))
        vector2 = np.array(mat['descriptors2'][descriptor]).reshape((-1, 1))
        distance = dist2(np.array(vector1), np.array(vector2)).mean()
        if distance < .0075:
            matches.append(descriptor)
            break
print len(matches)

fig=plt.figure()
bx=fig.add_subplot(111)
bx.imshow(im2)
coners = displaySIFTPatches(mat['positions2'][matches,:], mat['scales2'][matches,:], mat['orients2'][matches,:])

for j in range(len(coners)):
    bx.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1.5)
    bx.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1.5)
    bx.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1.5)
    bx.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1.5)
bx.set_xlim(0, im2.shape[1])
bx.set_ylim(0, im2.shape[0])
plt.gca().invert_yaxis()
plt.show()
