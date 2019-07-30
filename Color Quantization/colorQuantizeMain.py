import numpy as np
import matplotlib.pyplot as plt

from quantizeRGB import quantizeRGB
from quantizeHSV import quantizeHSV
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists

k = 3
img = plt.imread("fish.jpg")

quantizedRGB, _ = quantizeRGB(img, k)
quantizedHSV, _ = quantizeHSV(img, k)
ssdRGB = computeQuantizationError(img, quantizedRGB)
ssdHSV = computeQuantizationError(img, quantizedHSV)
# getHueHists(img, k)


# Answer sheet results
plt.imshow(quantizedRGB)
plt.title('Quantized RGB')
plt.show()

plt.imshow(quantizedHSV)
plt.title('Quantized HSV')
plt.show()

print(int(ssdRGB))

print(int(ssdHSV))
