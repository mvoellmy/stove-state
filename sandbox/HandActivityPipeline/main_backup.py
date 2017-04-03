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

def handTrajectory(segmented, segmented_old, p0, frame, mask, condition):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    pts_good = []
    if segmented_old.size == 0 or condition < 100:
        p0 = cv2.goodFeaturesToTrack(segmented, mask = None, **feature_params)
        mask = np.zeros_like(frame)
    else:
        p1, st, err = cv2.calcOpticalFlowPyrLK(segmented_old, segmented, p0, None, **lk_params)
        good_new = p1[st==1]
        good_old = p0[st==1]
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        p0 = good_new.reshape(-1,1,2)
        pts_good = p1[st==1]
    condition = condition + 1
    return p0, mask, condition, pts_good
    
   

#%% Murase dataset images
mypath='../../data/data1/boiled_egg_1/image_jpg'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
background = cv2.imread('../../data/data1/boiled_egg_1/image_jpg/image_00000001410.jpg')

segmented_old = np.array([])
p0 = np.array([])
mask = np.array([])
color = np.random.randint(0,255,(100,3))
condition = 0

text_file = open("trajectory.txt", "w")
img = cv2.imread( join(mypath,onlyfiles[0]) )
new_img = np.zeros(img.shape)
count = 0
for n in range(0, len(onlyfiles)):
    frame = cv2.imread( join(mypath,onlyfiles[n]) )
    # Hand segmentation
    segmented = segmentHand(frame)
    
    # Hand trajectory tracking
    p0, mask, condition, pts = handTrajectory(segmented, segmented_old, p0, frame, mask, condition)
    
    idx = 0
    if count > 100:
        new_img = cv2.line(new_img, (pts[idx,0],pts[idx,1]),(p0_old[idx,0],p0_old[idx,1]), 100, 2)
    p0_old = pts
    count += 1
    
    text_file.write("%s\t" % p0[0])
    text_file.write("%s\n" % p0[1])
    segmented_old = segmented

    # Display images
    cv2.imshow("Tracjectory", new_img)
    cv2.imshow("Segmented", segmented)
    k = cv2.waitKey(30)
    if k == 27: # Exit by pressing escape-key
        break

text_file.close()   
cv2.destroyAllWindows()

#%% Our video recording
cap = cv2.VideoCapture("../../data/place_noodles.mp4")

segmented_old = np.array([])
p0 = np.array([])
mask = np.array([])
color = np.random.randint(0,255,(100,3))
condition = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    
    # Hand segmentation
    segmented = segmentHand(frame)
    
    # Hand trajectory tracking
    p0, mask, condition, pts = handTrajectory(segmented, segmented_old, p0, frame, mask, condition)
    segmented_old = segmented
    
    # Display images
    cv2.imshow("Segmented", segmented)
    k = cv2.waitKey(30)
    if k == 27: # Exit by pressing escape-key
        break
    
cv2.destroyAllWindows()