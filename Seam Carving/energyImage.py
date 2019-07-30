import numpy as np
from skimage import color
import matplotlib.pyplot as plt

from scipy import ndimage

def energyImage(img):

    sobelX = [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]

    sobelY = [[1, 2, 1],
              [0, 0, 0],
              [-1, -2, -1]]

    img = color.rgb2gray(img)

    out = np.zeros_like(img)

    image_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    image_padded[1:-1, 1:-1] = img

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            dx = (sobelX * image_padded[y:y + 3, x:x + 3]).sum()
            dy = (sobelY * image_padded[y:y + 3, x:x + 3]).sum()
            out[y, x] = abs(dx) + abs(dy)

    plt.imshow(out)
    plt.show()
    return out


