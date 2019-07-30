import glob
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
from computeMHI import computeMHI
from huMoments import huMoments
from predictAction import predictAction
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# load data
# each row is a hu
hus = np.load('huVectors.npy')
labels = [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4
cm = np.zeros((5, 5))
actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']


for i in range(20):
    testVec = hus[i]
    trainData = np.delete(hus, i, axis=0)
    trainLabels = np.delete(labels, i, axis=0)
    prediction = predictAction(testVec, trainData, trainLabels)
    cm[i/4, prediction-1] = cm[i/4, prediction-1] + 1

plot_confusion_matrix(cm, actions, normalize=True)