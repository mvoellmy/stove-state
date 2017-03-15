from os import listdir
from os.path import isfile, join
import numpy as np
import cv2


def segmentHand(frame):
    frame = frame.astype(int)
    height, width = frame.shape[:2]
    threshold = -50
    r_g = frame[:,:,0] - frame[:,:,1]
    r_b = frame[:,:,0] - frame[:,:,2]
    lowest = cv2.min(r_g, r_b)
    # Compare lowest > threshold
    return cv2.compare(lowest, threshold, cmpop=1)

#%% Murase dataset images
mypath='../../data/data1/boiled_egg_1/image_jpg'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    frame = cv2.imread( join(mypath,onlyfiles[n]) )
    segmented = segmentHand(frame)

    cv2.imshow("Segmented", segmented)
    k = cv2.waitKey(30)
    if k == 27: # Exit by pressing escape-key
        break
    
cv2.destroyAllWindows()

#%% Our video recording
cap = cv2.VideoCapture("../../data/place_noodles.mp4")
while (cap.isOpened()):
    ret, frame = cap.read()
    
    segmented = segmentHand(frame)

    cv2.imshow("Segmented", segmented)
    k = cv2.waitKey(30)
    if k == 27: # Exit by pressing escape-key
        break
    
cv2.destroyAllWindows()