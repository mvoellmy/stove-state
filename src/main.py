
import configparser

from GestureRecognizer import *

config = configparser.ConfigParser()
config.read('../cfg/cfg.txt')
path_videos = config.get('paths', 'videos')

path_video = path_videos + '/I_begg/I_2017-04-06-20_08_45_begg.mp4'
cap = cv2.VideoCapture(path_video)

objectG = GestureRecognizer()
while (cap.isOpened()):

    ret, frame = cap.read()

    hand_in_frame = objectG.process_frame(frame)
    print(hand_in_frame)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == 27:  # Exit by pressing escape-key
        break