import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def quantizeHSV(orig_img, k):

    # Load Image and transform to a 2D numpy array.
    img_rgb = np.copy(orig_img)
    img_hsv = rgb_to_hsv(img_rgb)

    w, h, d = np.shape(img_hsv)
    image_array = np.reshape(img_hsv, (w * h, d))

    hue_array = image_array[:, 0]
    kmeans = KMeans(n_clusters=k)

    # Get labels for all points
    labels = kmeans.fit_predict(hue_array.reshape(-1, 1))

    means = kmeans.cluster_centers_
    image_out = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            hue = means[labels[label_idx]]
            image_out[i][j] = [hue, img_hsv[i, j, 1], img_hsv[i, j, 2]]
            label_idx += 1

    image_out = hsv_to_rgb(image_out)
    image_out = np.array(image_out, dtype='uint8')
    return image_out, means
