import numpy as np
import matplotlib.pyplot as plt


def displaySeam(img, seam, type):
    out = np.copy(img)
    if type == 'VERTICAL':
        for y in range(img.shape[0]):
            out[y, seam[y]] = [255, 0, 0]

    if type == 'HORIZONTAL':
        for x in range(img.shape[1]):
            out[seam[x], x] = [255, 0, 0]

    plt.imshow(out)
    plt.show()
