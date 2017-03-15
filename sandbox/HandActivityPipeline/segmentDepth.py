from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

#%%
mypath='../../data/data1/boiled_egg_1/depth_8bit'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
    thresh = cv2.inRange(images[n][:,:,0], 150, 254)
    cv2.imshow("Image", thresh)
    k = cv2.waitKey(5)
    if k == 27: # Exit by pressing escape-key
        break
    
cv2.destroyAllWindows()
    
#%%
mypath = '../../data/HandGestures/subject1_dep/K_person_1_backgroud_1_illumination_1_pose_1_actionType_2.avi'
cap = cv2.VideoCapture(mypath)

while (cap.isOpened()):
    ret, frame = cap.read()
    thresh = cv2.inRange(frame[:,:,0], 150, 240)
    cv2.imshow("Segmented", thresh)
    k = cv2.waitKey(5)
    if k == 27: # Exit by pressing escape-key
        break
    
cv2.destroyAllWindows()

