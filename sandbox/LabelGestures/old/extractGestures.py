
import configparser
import cv2
from os.path import join
import csv

config = configparser.ConfigParser()
config.read('../../cfg/cfg.txt')

path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')
file_name = 'I_20170419_232724_begg'
video_format = '.mp4'

cap = cv2.VideoCapture(join(path_videos, file_name + video_format))

gesture_data = []
path_gesture_labels = '../../../../Polybox/Shared/stove-state-data/ssds/gesture_labels/'
with open(join(path_gesture_labels, file_name + ".csv"), 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ')
    for row in csv_reader:
        try:
            gesture_data.append(row)
        except IndexError:
            gesture_data.append(-1)

while (cap.isOpened()):
    ret, frame = cap.read()
    dim = frame.shape

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    cv2.resizeWindow("Frame", int(dim[1] / 2), int(dim[0] / 2))
    k = cv2.waitKey(1)
    if k == 27:  # Exit by pressing escape-key
        break

cv2.destroyAllWindows()