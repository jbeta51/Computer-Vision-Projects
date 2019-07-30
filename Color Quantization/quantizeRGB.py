import numpy as np
from sklearn.cluster import KMeans


def quantizeRGB(orig_img, k):

    # Load Image and transform to a 2D numpy array.
    img = np.copy(orig_img)
    w, h, d = np.shape(img)
    image_array = np.reshape(img, (w * h, d))

    kmeans = KMeans(n_clusters=k).fit(image_array)

    # Get labels for all points
    labels = kmeans.predict(np.reshape(img, (w * h, d)))

    means = kmeans.cluster_centers_.astype(int)
    image_out = np.zeros((w, h, d), dtype="uint8")
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image_out[i][j] = means[labels[label_idx]]
            label_idx += 1

    return image_out, means
