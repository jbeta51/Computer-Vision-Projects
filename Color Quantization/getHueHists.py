import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from sklearn.cluster import KMeans


def getHueHists(img, k):
    hues = rgb_to_hsv(img)[:, :, 0]
    hues = np.ravel(hues)
    histEqual = plt.hist(hues, bins=k)
    plt.title('Equally Spaced Hue Hist')
    plt.show()

    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(hues.reshape(-1, 1))
    histClustered = plt.hist(labels, bins=k)
    plt.title('Cluster Membership')
    plt.show()

    return histEqual, histClustered
