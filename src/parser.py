import cv2
import configparser
import ast
import numpy as np
import csv
import os.path

# Options
video_type = '.mp4'
stove_type = 'I'

# plates_of_interest = np.array([0, 1, 0, 0])  # Ians kitchen
plate_of_interest = 4

# Read Config data
config = configparser.ConfigParser()
config.read('../cfg/cfg.txt')

# Read corners and reshape them into 2d-Array
corners = np.reshape(ast.literal_eval(config.get("corners", "coordinates")), (-1, 4))

# Read dataset
path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')

has_pan = np.zeros((1, 4))


# Import & parse dataset into labels and frames
list_videos = []
list_videos = [f for f in os.listdir(path_videos) if os.path.isfile(os.path.join(path_videos, f))
                                                     and video_type in f
                                                     and f.startswith(stove_type)]

# patch_width = int(np.floor((corners[1, 0] - corners[0, 0]) / 2))
# patch_height = int(np.floor((corners[2, 1] - corners[0, 1]) / 2))
# frames = np.zeros((np.count_nonzero(plates_of_interest), patch_height, patch_width))

for video in list_videos:
    # Single Video:
    # path_video = path_videos + '2017-03-31-05_01_45_egg_boiled.mp4'

    # find corresponding labeling file
    label_file = video.replace(video_type, ".csv")
    path_label = path_labels + label_file




    # Read file
    path_video = path_videos + video
    cap = cv2.VideoCapture(path_video)
    frame_rate = cap.get(5)
    nr_of_frames = cap.get(7)

    frame_id = 1
    frame_time = frame_id*frame_rate

    pan_labels = np.zeros(int(nr_of_frames))  # 0: no pan, 1: pan, 2: cover

    with open(path_label, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', )
        for row in csv_reader:
            if str(plate_of_interest) == row[1]:
                if 'pan' in row[2] and 1 == int(row[3]):
                    pan_labels[int(frame_time * float(row[0])):] = 1
                    print(int(frame_time * float(row[0])))

                elif 'pan' in row[2] and 1 == int(row[3]):
                    pan_labels[int(frame_time * float(row[0])):] = 0
                    print(int(frame_time * float(row[0])))

                elif 'cover' in row[2] and 1 == int(row[3]):
                    pan_labels[int(frame_time * float(row[0])):] = 2
                    print(int(frame_time * float(row[0])))

                elif 'cover' in row[2] and 0 == int(row[3]):
                    pan_labels[int(frame_time * float(row[0])):] = 1
                    print(int(frame_time * float(row[0])))

    print("Labeling done")

    # for frames in video
    while cap.isOpened():
        ret, frame = cap.read()

        # TODO for multiple plates
        # for i, k in enumerate(plates_of_interest):
        #     if k > 0:
        #         cv2.rectangle(frame, tuple(corners[i, 0:2]), tuple(corners[i, 2:4]), 255)
        cv2.rectangle(frame, tuple(corners[plate_of_interest-1, 0:2]), tuple(corners[plate_of_interest-1, 2:4]), 255)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        # split plates
        patch = frame[corners[plate_of_interest-1, 1]:corners[plate_of_interest-1, 3], corners[plate_of_interest-1, 0]:corners[plate_of_interest-1, 2]]
        patch_title = 'Label: ' + str(int(pan_labels[frame_id]))
        cv2.imshow(patch_title, patch)
        # print(int(pan_labels[frame_id]))

        # extract features and save labels
        frame_id = frame_id + 1


# Extract Features from images