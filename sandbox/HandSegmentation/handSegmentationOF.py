
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import measure

cap = cv2.VideoCapture("../../data/place_noodles.mp4")
#cap = cv2.VideoCapture("../../data/test_dark.mp4")
#cap = cv2.VideoCapture("../../../../Polybox/Shared/stove-state-data/ssds/gestures/remove_egg.mp4")
#cap = cv2.VideoCapture("../../data/salmon_noodles.mp4")

prevgray = []
while (cap.isOpened()):
    ret, frame = cap.read()

    # Hand segmentation
    frame_int = frame.astype(int)
    height, width = frame.shape[:2]

    """
    R = frame_int[:, :, 0].astype(float)
    G = frame_int[:, :, 1].astype(float)
    B = frame_int[:, :, 2].astype(float)

    norm = np.zeros(frame.shape,dtype=float)

    RG = cv2.add(R,G)
    RGB = cv2.add(RG,B)
    RGB = RGB.astype(np.float)

    a = cv2.divide(R,RGB)
    b = cv2.divide(G, RGB)
    c = cv2.divide(B, RGB)
    norm[:,:,0] = cv2.divide(R,RGB)
    norm[:,:,1] = cv2.divide(G, RGB)
    norm[:,:,2] = cv2.divide(B, RGB)

    norm = norm*255
    norm = norm.astype(np.uint8)
    """


    imgYCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    Y = imgYCC[:, :, 0]
    Cr = imgYCC[:, :, 1]
    Cb = imgYCC[:, :, 2]
    skin_ycrcb_mint = np.array((0, 140, 100))
    skin_ycrcb_maxt = np.array((255, 165, 120))
    skin_ycrcb = cv2.inRange(imgYCC, skin_ycrcb_mint, skin_ycrcb_maxt)

    flow = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prevgray != []:
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    flow = flow.astype(np.uint8)

    # Display images
    cv2.imshow("Frame", skin_ycrcb)
    cv2.imshow("Normalized", gray)
    k = cv2.waitKey(1)
    if k == 27:  # Exit by pressing escape-key
        break

cv2.destroyAllWindows()