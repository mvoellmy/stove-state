
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
plt.ion()

#######################################################################################
# Function Definitions

def segment_hand(frame):
    frame = frame.astype(int)
    height, width = frame.shape[:2]
    threshold = -50
    r_g = frame[:,:,0] - frame[:,:,1]
    r_b = frame[:,:,0] - frame[:,:,2]
    lowest = cv2.min(r_g, r_b)
    # Compare lowest > threshold
    segmented_inv = cv2.compare(lowest, threshold, cmpop=1)
    segmented = (segmented_inv == np.zeros(segmented_inv.shape))*255
    return segmented.astype(np.uint8)

def connected_components(segmented):
    output = cv2.connectedComponentsWithStats(segmented)
    num_components = output[0]
    img = output[1]
    idx_max = 0
    val_max = 0
    for i in range(1, 10):
        val = np.sum(np.sum(img == np.ones(img.shape) * i) * 1)
        if val > val_max:
            val_max = val
            idx_max = i

    # img = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # all_labels = measure.label(segmented)
    S = (img == np.ones(img.shape) * idx_max) * 1
    S = cv2.multiply(S, 255)
    S = S.astype(np.uint8)
    return S

def PCA_direction(segmented_CC, C):
    y, x = np.nonzero(segmented_CC)
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    evec1, evec2 = evecs[:, sort_indices]
    x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evec2
    scale = 60
    a = int(x_v1*-scale*2 + C[1])
    c = int(x_v1*scale*2 + C[1])
    b = int(y_v1*-scale*2 + C[0])
    d = int(y_v1*scale*2 + C[0])
    return a, b, c, d

######################################################################################
# Main Program

cap = cv2.VideoCapture("../../data/place_noodles.mp4")
cap = cv2.VideoCapture("../../../../Polybox/Shared/stove-state-data/ssds/gestures/place_schnitzel_1.mp4")

gray_old = []
while (cap.isOpened()):
    ret, frame = cap.read()
    dim = frame.shape

    # Color Segmentation --------------------------------------------
    segmented = segment_hand(frame)


    # Frame Subtraction ---------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    differential = np.zeros(frame.shape[0:2])
    if gray_old != []:
        differential = cv2.subtract(gray,gray_old)
    gray_old = gray

    differential = (differential > 15) * 255
    differential = differential.astype(np.uint8)

    segmented_final = cv2.multiply(segmented/255, differential/255)*255
    segmented_final = segmented_final.astype(np.uint8)


    # Connected Components -----------------------------------------
    segmented_final = connected_components(segmented)


    # Compute centroid ---------------------------------------------
    centroid = ndimage.measurements.center_of_mass(segmented_final)
    if centroid[0] != centroid[0]:
        centroid = (0,0)


    # PCA for Hand Orientation -------------------------------------
    if np.sum(np.sum(segmented_final)) > 0:
        a, b, c, d = PCA_direction(segmented_final, centroid)
        frame = cv2.line(frame, (a, b), (c, d), (0, 0, 255), 10)

    frame = cv2.circle(frame, (int(centroid[1]), int(centroid[0])), 10, (0, 255, 0), -1)

    # plt.figure(1)
    # plt.imshow(frame)
    # plt.show()
    # plt.pause(0.0005)

    # Display Images
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    cv2.resizeWindow("Frame", int(dim[1]/2), int(dim[0]/2))
    cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)
    cv2.imshow("Segmented", segmented_final)
    cv2.resizeWindow("Segmented", int(dim[1] / 2), int(dim[0] / 2))
    k = cv2.waitKey(1)
    if k == 27:  # Exit by pressing escape-key
       break

cv2.destroyAllWindows()

#####################################################################################