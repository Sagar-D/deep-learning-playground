import numpy as np


def generate_augmented_image_data(X, Y):
    """
    Generate augmented image data by flipping/roatting the existing image dataset

    Arguments :
        X => numpy array of image dataset of shape (m,i,j,x)
        Y => numpy array of image labels of shape (m,1)
            where,
                m = total number of training samples in the training set
                i,j = dimension of each image
                x = pixel size (3 for RGB, 1 for B/W)

    Return :
        augmented_X => numpy array of image dataset with original and augmented images
        augmented_Y => numpy array of image labels with original and augmented images
    """

    augmented_X = []
    augmented_Y = []

    for i in range(Y.shape[0]):
        augmented_X.append(X[i])
        augmented_Y.append(Y[i][0])
        augmented_X.append(np.flip(X[i], axis=1))
        augmented_Y.append(Y[i][0])
        if Y[i][0] == 1:
            augmented_X.append(np.rot90(X[i], k=1))
            augmented_Y.append(Y[i][0])
            augmented_X.append(np.rot90(X[i], k=-1))
            augmented_Y.append(Y[i][0])

    augmented_X = np.array(augmented_X)
    augmented_Y = np.array(augmented_Y)
    augmented_Y = augmented_Y.reshape(augmented_Y.shape[0], 1)

    return augmented_X, augmented_Y
