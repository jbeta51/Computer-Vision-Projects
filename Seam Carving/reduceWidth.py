from cumulativeEnergyMap import cumulativeEnergyMap
from verticalSeam import verticalSeam

import numpy as np
import matplotlib.pyplot as plt


def reduceWidth(img, energyImg):

    cum_vert = cumulativeEnergyMap(energyImg, 'VERTICAL')
    vert_seam = verticalSeam(cum_vert)

    out = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), dtype='uint8')
    newEnergy = np.zeros((energyImg.shape[0], energyImg.shape[1] - 1))

    for y in range(img.shape[0]):
        row = img[y]
        newRow = row[np.arange(len(row)) != vert_seam[y]]

        energyRow = energyImg[y]
        newEnergyRow = energyRow[np.arange(len(energyRow)) != vert_seam[y]]

        out[y] = newRow
        newEnergy[y] = newEnergyRow

    return out, newEnergy
