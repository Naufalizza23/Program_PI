import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features(image):
    image = cv2.resize(image, (128, 128))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    # Histogram RGB
    for channel in cv2.split(image):
        hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
        features.extend(hist.flatten())

    # Histogram HSV
    for channel in cv2.split(hsv):
        hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
        features.extend(hist.flatten())

    # Fitur tekstur GLCM
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        features.append(graycoprops(glcm, prop)[0, 0])

    return np.array(features)
