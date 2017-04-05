import configparser
import numpy as np
import cv2
from os.path import join
import time
import scipy
from scipy import signal

config = configparser.ConfigParser()
config.read('../../cfg/cfg.txt')

path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')
file_name = 'I_2017-04-02-10_14_19_schnitzel'
video_format = '.mp4'

#path_videos = ''
#path_labels = ''
#file_name = 'schnitzel_shorter'

cap = cv2.VideoCapture(join(path_videos, file_name + video_format))

label_file = open(join(path_labels, file_name + "_gesture.csv"), "w")
time_start = time.time()
count = 0
labels = []
while (cap.isOpened()):
    count += 1
    ret, frame = cap.read()
    if frame == None:
        break
    dim = frame.shape

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    cv2.resizeWindow("Frame", int(dim[1] / 2), int(dim[0] / 2))
    k = cv2.waitKey(50)
    if k == 27:  # Exit by pressing escape-key
        break
    if k >= 48 and k <= 57:
        k = k - 48
    else:
        k = -1
    print(k)

    row = [count, k]
    labels.append(row)

labels = np.array(labels)
labels[:,1] = scipy.signal.medfilt(labels[:,1],3) # Apply median filter

for row in labels:
    label_file.write(str(row[0])+" ")
    label_file.write(str(row[1]) + "\n")
label_file.close()
cv2.destroyAllWindows()