
import configparser
from os.path import join
import cv2
from pipeline import pipeline

config = configparser.ConfigParser()
config.read('../../cfg/cfg.txt')

path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')
path_gestures = '../../../../Polybox/Shared/stove-state-data/ssds/gestures/'
video_format = '.mp4'

path_features = 'gesture_features'
path_recording = ''
file_names = ['I_2017-04-06-20_08_45_begg',
             'I_2017-04-13-21_26_55_begg',
             'I_20170419_232724_begg',
             'I_raspivid_20170421_begg',
             'I_20170424_210116_begg',
             'I_20170428_224946_begg',
             'I_20170430_210819_begg',
             'I_20170503_232838_begg']
# file_names = ['I_20170425_205126_scegg',
#               'I_20170427_212553_scegg']
file_names = ['I_20170501_212055_segg',
              'I_20170502_212256_segg',
              'I_20170503_234946_segg']
file_names = [ file_names[2] ]
num_files = len(file_names)
num_gestures = 8

for file_name in file_names:
    for gesture_num in range(1,num_gestures+1):
        # gesture_num = 6
        recipe_name = file_name.rsplit("_", 1)[-1].split(".")[0]  # Takes the word after the last underscore
        path_video = join(path_videos, file_name + video_format)
        path_gesture = join(path_gestures, recipe_name, '{}'.format(gesture_num), file_name + '.h264')
        cap_gesture = cv2.VideoCapture(path_gesture)
        cap_video = cv2.VideoCapture(path_video)

        if cap_gesture.isOpened():
            path_feature_file = join(path_features, recipe_name, "{}_".format(gesture_num) + file_name + "_features.csv")
        else:
            path_feature_file = join(path_features, recipe_name, "{}_".format(gesture_num) + file_name + "_features_false.csv")
        path_video_file = join(path_recording, "{}_".format(gesture_num) + file_name + '.avi')
        # path_feature_file=[]
        path_video_file=[]

        pipeline(cap_gesture, cap_video, path_feature_file, path_video_file)