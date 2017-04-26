
import configparser
from os.path import join
import csv
import os

config = configparser.ConfigParser()
config.read('../../cfg/cfg.txt')

path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')
file_name = 'I_20170425_205126_scegg'
video_format = '.mp4'

path_frame_labels = '../../../../Polybox/Shared/stove-state-data/ssds/frame_labels/'
path_gestures = '../../../../Polybox/Shared/stove-state-data/ssds/gestures/'

gesture_data = []
with open(join(path_frame_labels, file_name + ".csv"), 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ')
    for row in csv_reader:
        try:
            gesture_data.append(row)
        except IndexError:
            gesture_data.append(-1)

num_gestures = len(gesture_data)

path_video = join(path_videos, file_name + video_format)

for i in range(0,num_gestures):
    recipe_name = file_name.rsplit("_",1)[-1].split(".")[0] # Takes the word after the last underscore
    path_gesture = join(path_gestures, recipe_name, "{}".format(i+1), file_name + ".h264")
    frame_start = gesture_data[i][0]
    frame_end = gesture_data[i][1]
    print("ffmpeg -i {} -vf trim=start_frame={}:end_frame={} {}".format(path_video, frame_start, frame_end, path_gesture))
    os.system("ffmpeg -i {} -vf trim=start_frame={}:end_frame={} {}".format(path_video, frame_start, frame_end, path_gesture))