import numpy as np


def computeQuantizationError(origImg, quantizedImg):
    return np.linalg.norm(np.sum(origImg - quantizedImg, axis=2))**2
