import configparser
import numpy as np
import scipy
from scipy import signal
import cv2
from os.path import join
import time
import csv

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

labels = []
with open(join(path_labels, file_name + "_gesture.csv"), 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ')
    for row in csv_reader:
        try:
            labels.append(int(row[1]))
        except IndexError:
            labels.append(-1)
time_start = time.time()
count = 0
while (cap.isOpened()):
    count += 1
    ret, frame = cap.read()
    dim = frame.shape


    #print(labels[count])
    if int(labels[count]) == 4:
        gesture = "Place"
    elif int(labels[count]) == 5:
        gesture = "Flip"
    elif int(labels[count]) == 6:
        gesture = "Remove"
    elif int(labels[count]) == 2:
        gesture = "Stirr"
    elif int(labels[count]) == 8:
        gesture = "Season"
    else:
        gesture = "No gesture"
    frame = cv2.putText(frame, gesture, (50,100), cv2.FONT_HERSHEY_DUPLEX, 3,(0,255,255),5)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    cv2.resizeWindow("Frame", int(dim[1] / 2), int(dim[0] / 2))
    k = cv2.waitKey(1)
    if k == 27:  # Exit by pressing escape-key
        break

cv2.destroyAllWindows()