from energyImage import energyImage
from reduceHeight import reduceHeight


import matplotlib.pyplot as plt


prague = plt.imread("inputSeamCarvingPrague.jpg")
prague_energy = energyImage(prague)

for i in range(100):
    prague, prague_energy = reduceHeight(prague, prague_energy)

plt.imshow(prague)
plt.show()
