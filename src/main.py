
import configparser

# Own Libraries
from utils_gesture import *
from food_recognizer import *
from gesture_recognizer import *
from particle_filter import *

import time

config = configparser.ConfigParser()
config.read('../cfg/cfg.txt')
path_videos = config.get('paths', 'videos')

# Choose which stove and plate is observed
plate_of_interest = 'I_4'

if plate_of_interest == 'I_4':
    path_video = path_videos + '/I_begg/I_20170516_212934_multiple.mp4'
    path_video = path_videos + '/I_begg/I_2017-04-06-20_08_45_begg.mp4'
    path_video = '/Users/miro/Polybox/Shared/ssds_ian/pan_detect/test_videos/pot_positions.mp4'
    path_video = path_videos + '/I_begg/demo_begg.mov'

elif plate_of_interest == 'I_2':
    path_video = path_videos + '/I_scegg/I_20170430_213149_scegg.mp4'
    path_video = path_videos + '/I_segg/I_20170504_221703_segg.mp4'
    path_video = '/Users/miro/Polybox/Shared/stove-state-data/ssds/pan_detect/test_videos/segg_short.mov'
    path_video = '/Users/miro/Polybox/Shared/ssds_ian/pan_detect/test_videos/pan_positions.mp4'
    path_video = '/Users/miro/Polybox/Shared/stove-state-data/ssds/pan_detect/test_videos/scegg_test_2.mp4'
    path_video = path_videos + '/I_scegg/I_20170425_205126_scegg.mp4'  # this is gud
    path_video = path_videos + '/I_scegg/I_20170427_212553_scegg.mp4'

elif plate_of_interest == 'M_2':
    path_video = path_videos + '/M_scegg/M_2017-05-06-09_15_26_scegg.mp4'

elif plate_of_interest == 'M_4':
    path_video = path_videos + '/M_begg/M_2017-04-22-14_15_50_begg.mp4'

cap = cv2.VideoCapture(path_video)

food_rec = FoodRecognizer(plate_of_interest=plate_of_interest)
gesture_rec = GestureRecognizer()
p_filter = ParticleFilter(200)

# Playback Options
_start_frame = 770   # pan positions
_start_frame = 0
_start_frame = 300  # segg to scegg

_end_frame = -1
_frame_rate = 1  # Only process every 'n'th frame

# Plot Options
_plot_segmentation = False

frame_id = 0
food_rec_time = -2
food_check_interval = 25
curr_food_rec_time = math.floor(food_rec_time/_frame_rate)

# Variable Init
pan_state_dist = []

while cap.isOpened():

    frame_id += 1

    start_time = time.time()

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
        curr_food_rec_time = math.floor(food_rec_time/_frame_rate)
        print('Gesture:\t{}'.format(gesture))

    # if True:
    if curr_food_rec_time != 0:
        curr_food_rec_time -= _frame_rate

        # Recognize Food
        pan_label_name, food_label_name, pan_label_id, food_label_id = food_rec.process_frame(frame)

        p_filter.update_particles(gesture)
        p_filter.update_weights(pan_label_name)
        pan_state_dist = p_filter.count_particles()

        #if 'pan' in pan_label_name:
        if True:
            center, axes, phi = food_rec.get_pan_location()
            cv2.ellipse(frame, tuple(map(int, center)), tuple(map(int, axes)),
                        int(-phi * 180 / pi), 0, 360, (0, 0, 255), thickness=5)

    # Output Results
    food_label_dict = {'begg': 'boiled egg',
                       'segg': 'sunny egg',
                       'scegg': 'scrambled egg',
                       'empty': 'empty',
                       'water': 'water'}

    cv2.putText(frame, str(pan_label_name), (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0), 5)
    cv2.putText(frame, str(food_label_dict[food_label_name]), (0, 600), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0), 5)
    cv2.putText(frame, str(gesture), (0, 800), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 100, 255), 5)
    cv2.putText(frame, str(frame_id), (0, 1200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
    cv2.putText(frame, 'Plate: {} Pan:{} Lid:{}'.format(pan_state_dist[0],
                                                        pan_state_dist[1],
                                                        pan_state_dist[2]),
                (600, 1200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


    # cv2.namedWindow("Frame", 2)
    # cv2.resizeWindow("Frame", 640, 480)
    # cv2.imshow("Frame", frame)
    cv2.imwrite("frame_{}.jpg".format(frame_id), frame)

    if gesture != []:
        k = cv2.waitKey(2000)
    else:
        k = cv2.waitKey(1)

    if k == 27:  # Exit by pressing escape-key
        break

    print('Frame time: {}'.format(time.time() - start_time))

print('Program finished successfully')
