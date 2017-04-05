from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from numpy import linalg as LA


def segmentHand(frame):
    frame = frame.astype(int)
    height, width = frame.shape[:2]
    threshold = -50
    r_g = frame[:,:,0] - frame[:,:,1]
    r_b = frame[:,:,0] - frame[:,:,2]
    lowest = cv2.min(r_g, r_b)
    # Compare lowest > threshold
    return cv2.compare(lowest, threshold, cmpop=1)

def handTrajectory(segmented, segmented_old, p0, frame, mask, condition, p0_idx):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    vel = []
    if condition%100 == 0:
        mask = np.zeros_like(frame)
        
    if segmented_old.size == 0 or condition < 100:
        p0 = cv2.goodFeaturesToTrack(segmented, mask = None, **feature_params)
        p0_idx = np.arange(0, p0.shape[0])
        mask = np.zeros_like(frame)
        
    elif p0.shape[0] != 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(segmented_old, segmented, p0, None, **lk_params)
        
        inliers = st==1
        good_new = p1[inliers]
        good_old = p0[inliers]
        
        p0_idx = p0_idx[inliers[:,0]]
        vel = LA.norm(good_new - good_old, axis=1)
        inliers2 = vel > 2.5
        good_new = good_new[inliers2]
        good_old = good_old[inliers2]
        p0_idx = p0_idx[inliers2]
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d),(0,0,255) , 2) #color[i].tolist()
            frame = cv2.circle(frame,(a,b),5,(0,0,255),-1)
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        p0 = good_new.reshape(-1,1,2)
    
    if p0.shape[0] < 20:
        p_add = cv2.goodFeaturesToTrack(segmented, mask = None, **feature_params)
        max_idx = np.max(p0_idx)
        p_idx_add = np.arange(max_idx,p_add.shape[0]+max_idx)
        p0 = np.concatenate((p0,p_add),0)
        p0_idx = np.concatenate((p0_idx,p_idx_add),0)
        
    
    return p0, mask, p0_idx, vel
    
   

#%% Murase dataset images
mypath='../../data/data1/scrambled_egg_1/image_jpg'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
background = cv2.imread('../../data/data1/boiled_egg_1/image_jpg/image_00000001410.jpg')

segmented_old = np.array([])
p0 = np.array([])
mask = np.array([])
color = np.random.randint(0,255,(100,3))

text_file = open("trajectory.txt", "w")
img = cv2.imread( join(mypath,onlyfiles[0]) )
new_img = np.zeros(img.shape)
count = 0
p0_idx = []

for n in range(0, len(onlyfiles)):
    
    frame = cv2.imread( join(mypath,onlyfiles[n]) )
    
    # Hand segmentation
    segmented = segmentHand(frame)
    
    # Hand trajectory tracking
    idx = 81
    p0, mask, p0_idx, vel = handTrajectory(segmented, segmented_old, p0, frame, mask, count, p0_idx)
    count += 1
    segmented_old = segmented
    print(p0.shape)
    
#    s_idx = np.where(p0_idx==idx)[0][0]
#    print(s_idx)
#    
#    if count > 100:
#        a, b = p0[s_idx,0,0], p0[s_idx,0,1]
#        c, d = p0_old[s_idx,0,0],p0_old[s_idx,0,0]
#        #new_img = cv2.line(new_img, (a,b),(c,d), 155, 1)
#        new_img = cv2.circle(new_img,(a,b),5,155,-1)
#        text_file.write("%s\n" % p0[s_idx].astype(np.int))
#        #text_file.write("%s\n" % p0[s_idx])
#    p0_old = p0

    # Display images
    cv2.imshow("Trajectory", new_img)
    cv2.imshow("Segmented", segmented)
    k = cv2.waitKey(30)
    if k == 27: # Exit by pressing escape-key
        break

text_file.close()   
cv2.destroyAllWindows()
