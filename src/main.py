
import configparser

# Own Libraries
from utils_gesture import *
from food_recognizer import *
from gesture_recognizer import *

config = configparser.ConfigParser()
config.read('../cfg/cfg.txt')
path_videos = config.get('paths', 'videos')

path_video = path_videos + '/I_scegg/I_20170427_212553_scegg.mp4'
path_video = '/Users/miro/Polybox/Shared/stove-state-data/ssds/pan_detect/test_videos/segg_short.mov'
path_video = '/Users/miro/Polybox/Shared/stove-state-data/ssds/pan_detect/test_videos/scegg_test_2.mp4'
path_video = path_videos + '/I_begg/I_2017-04-06-20_08_45_begg.mp4'

cap = cv2.VideoCapture(path_video)

food_rec = FoodRecognizer()
gesture_rec = GestureRecognizer()

# Playback Options
_start_frame = 200
_end_frame = -1
_frame_rate = 100  # Only process every 'n'th frame

# Plot Options
_plot_segmentation = True

frame_id = 0
food_rec_time = -1
curr_food_rec_time = math.floor(food_rec_time/_frame_rate)

while cap.isOpened():

    frame_id += 1

    # Read an process frame
    ret, frame = cap.read()

    if frame_id < _start_frame or (frame_id % _frame_rate != 0):
        continue
    elif frame_id + _start_frame > _end_frame > -1:
        break

    # Recognize Gesture
    gesture = gesture_rec.process_frame(frame)

    # if not gesture:
    if gesture != []:
        curr_food_rec_time = int(food_rec_time/_frame_rate)
        print('Gesture:\t{}'.format(gesture))

    # if True:
    if curr_food_rec_time != 0:
        curr_food_rec_time -= _frame_rate

        # Recognize Food
        pan_label_name, food_label_name = food_rec.process_frame(frame)

        if 'pan' in pan_label_name:
        # if True:
            center, axes, phi = food_rec.get_pan_location()
            cv2.ellipse(frame, tuple(map(int, center)), tuple(map(int, axes)),
                        int(-phi * 180 / pi), 0, 360, (0, 0, 255), thickness=5)

        # Output Results
        cv2.putText(frame, str(pan_label_name), (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0), 5)
        cv2.putText(frame, str(food_label_name), (0, 600), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0), 5)
        cv2.putText(frame, str(gesture), (0, 800), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 100, 255), 5)
        cv2.putText(frame, str(frame_id+_start_frame), (0, 1200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    cv2.namedWindow("Frame", 2)
    # cv2.resizeWindow("Frame", 640, 480)
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == 27:  # Exit by pressing escape-key
        break

print('Program finished successfully')
