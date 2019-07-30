from cumulativeEnergyMap import cumulativeEnergyMap
from horizontalSeam import horizontalSeam

import numpy as np
import matplotlib.pyplot as plt


def reduceHeight(img, energyImg):

    cum_hor = cumulativeEnergyMap(energyImg, 'HORIZONTAL')
    hor_seam = horizontalSeam(cum_hor)

    out = np.zeros((img.shape[0] - 1, img.shape[1], img.shape[2]), dtype='uint8')
    newEnergy = np.zeros((energyImg.shape[0] - 1, energyImg.shape[1]))

    for x in range(img.shape[1]):
        col = img[:, x]
        newCol = col[np.arange(len(col)) != hor_seam[x]]

        energyRow = energyImg[:, x]
        newEnergyRow = energyRow[np.arange(len(energyRow)) != hor_seam[x]]

        out[:, x] = newCol
        newEnergy[:, x] = newEnergyRow

    return out, newEnergy
