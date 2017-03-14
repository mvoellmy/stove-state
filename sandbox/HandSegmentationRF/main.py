
import numpy as np
import cv2

cap = cv2.VideoCapture("../../data/place_noodles.mp4")

while (cap.isOpened()):
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    S = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            R = int(frame[i][j][0])
            G = int(frame[i][j][1])
            B = int(frame[i][j][2])
            if min(R - G, R - B) > -50:
                S[i][j] = 1
            else:
                S[i][i] = 0
    cv2.imshow("Segmented", S)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)  