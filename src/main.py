
import configparser

# Own Libraries
from utils_gesture import *
from recognition import *

config = configparser.ConfigParser()
config.read('../cfg/cfg.txt')
path_videos = config.get('paths', 'videos')

path_video = path_videos + '/I_begg/I_2017-04-06-20_08_45_begg.mp4'
cap = cv2.VideoCapture(path_video)

recognizer = Recognition()

# Options
_start_frame = 0
_end_frame = 1
_frame_rate = 1  # Only process every 'n'th frame

frame_id = 0

while cap.isOpened():

    frame_id += 1

    if frame_id < _start_frame or (frame_id % _frame_rate != 0):
        continue
    elif frame_id - _start_frame > _end_frame:
        break

    # Read an process frame
    ret, frame = cap.read()

    pan_label_name, food_label_name = recognizer.process_frame(frame)

    if 'pan' in pan_label_name:
        center, axes, phi = recognizer.get_pan_location()


    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == 27:  # Exit by pressing escape-key
        break