import numpy as np
import cv2
import math

def segmentation_RGB(frame):
    frame = frame.astype(int)
    height, width = frame.shape[:2]
    threshold = -70
    r_g = frame[:,:,0] - frame[:,:,1]
    r_b = frame[:,:,0] - frame[:,:,2]
    lowest = cv2.min(r_g, r_b)
    # Compare lowest > threshold
    segmented_inv = cv2.compare(lowest, threshold, cmpop=1)
    segmented = (segmented_inv == np.zeros(segmented_inv.shape))*255
    return segmented.astype(np.uint8)

def segmentation_YCC(frame):
    # http://stackoverflow.com/questions/14752006/computer-vision-masking-a-human-hand
    imgYCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    Y = imgYCC[:, :, 0]
    Cr = imgYCC[:, :, 1]
    Cb = imgYCC[:, :, 2]
    skin_ycrcb_mint = np.array((0, 150, 0))
    skin_ycrcb_maxt = np.array((255, 255, 255))
    skin_ycrcb = cv2.inRange(imgYCC, skin_ycrcb_mint, skin_ycrcb_maxt)

    return skin_ycrcb

def segmentation_HSV(frame):
    # http://stackoverflow.com/questions/14752006/computer-vision-masking-a-human-hand
    skin_min = np.array([0, 0, 120], np.uint8)
    skin_max = np.array([20, 255, 255], np.uint8)
    gaussian_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    blur_hsv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)
    segmented = cv2.inRange(blur_hsv, skin_min, skin_max)

    return segmented

def connected_components(segmented):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented)

    # Loop through all components except the first one, because it is the background (biggest)
    max_area = 0
    idx_max = 1
    for i in range(1,ret):
        if stats[i][4] > max_area:
            max_area = stats[i][4]
            idx_max = i

    img = (labels == idx_max)*255
    img = img.astype(np.uint8)

    centroid = np.asarray(np.flip(centroids[idx_max],0))

    # Discard everything if the area is not big enough
    threshold_area = 20000
    validation = True
    if max_area < threshold_area:
        validation = False

    return img, centroid, validation

def PCA_direction(segmented_CC, centroid):
    # https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
    y, x = np.nonzero(segmented_CC)
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    evec1, evec2 = evecs[:, sort_indices]
    x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
    y_v2, x_v2 = evec2

    if abs(x_v1) > abs(y_v1) and np.sign(x_v1) == 1: # staehlii: cheat to avoid orientation jumping around
        y_v1 = - y_v1

    scale = 100
    x1 = int(x_v1*-scale + centroid[1])
    x2 = int(x_v1*scale + centroid[1])
    y1 = int(y_v1*-scale + centroid[0])
    y2 = int(y_v1*scale + centroid[0])

    return x1, y1, x2, y2

def extract_STF(data, N):
    num_frames = data.shape[0]
    step = int(num_frames / N)
    STF = np.zeros((2, N))
    idx = 0
    for f in range(0, N):
        STF[0,f] = round(np.interp(math.atan2(-data[idx, 2], data[idx, 3]), [-math.pi, math.pi], [0, 16]), 0)
        STF[1,f] = np.interp(data[idx, 4], [-math.pi, math.pi], [0, 16], 0)
        idx += step

    return STF