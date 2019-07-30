from energyImage import energyImage
from reduceWidth import reduceWidth

import matplotlib.pyplot as plt


prague = plt.imread("Cliffs.jpg")
prague_energy = energyImage(prague)

for i in range(100):
    prague, prague_energy = reduceWidth(prague, prague_energy)

plt.imshow(prague)
plt.show()
