import numpy as np


def cumulativeEnergyMap(img, mode):

    if mode == 'VERTICAL':
        img = np.rot90(img)

    out = np.zeros_like(img)
    for x in range(1, img.shape[1]):
        for y in range(img.shape[0]):
            if y == 0:
                out[y, x] = img[y, x] + min([out[y, x - 1], out[y + 1, x - 1]])
            elif y == img.shape[0] - 1:
                out[y, x] = img[y, x] + min([out[y, x - 1], out[y - 1, x - 1]])
            else:
                out[y, x] = img[y, x] + min([out[y, x - 1], out[y + 1, x - 1], out[y - 1, x - 1]])

    if mode == 'VERTICAL':
        out = np.rot90(out, 3)
    return out

