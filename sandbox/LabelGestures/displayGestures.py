import configparser
import numpy as np
import cv2
from os.path import join
import time
import csv

config = configparser.ConfigParser()
config.read('../../cfg/cfg.txt')

path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')

path_video = 'I_2017-04-02-10_14_19_schnitzel.mp4'
path_video = 'schnitzel_short.mp4'

cap = cv2.VideoCapture(join(path_videos, path_video))

# label_file = open("schnitzel_short.csv", "r")
# labels = label_file.read()
label_file = "schnitzel_short.csv"
labels = []
with open(label_file, 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ')
    for row in csv_reader:
        labels.append(row)
time_start = time.time()
count = 0
while (cap.isOpened()):
    count += 1
    ret, frame = cap.read()
    dim = frame.shape


    print(labels[count][2])

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


cv2.destroyAllWindows()