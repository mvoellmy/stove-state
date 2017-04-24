
import numpy as np
import cv2
from scipy import ndimage
import configparser
from os.path import join
import math

#######################################################################################
# Function Definitions

def segmentation_RGB(frame):
    frame = frame.astype(int)
    height, width = frame.shape[:2]
    threshold = -70
    r_g = frame[:,:,0] - frame[:,:,1]
    r_b = frame[:,:,0] - frame[:,:,2]
    lowest = cv2.min(r_g, r_b)
    # Compare lowest > threshold
    segmented_inv = cv2.compare(lowest, threshold, cmpop=1)
    segmented = (segmented_inv == np.zeros(segmented_inv.shape))*255
    return segmented.astype(np.uint8)

def connected_components(segmented):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented)

    # Loop through all components except the first one, because it is the background (biggest)
    max_area = 0
    idx_max = 1
    for i in range(1,ret):
        if stats[i][4] > max_area:
            max_area = stats[i][4]
            idx_max = i

    img = (labels == idx_max)*255
    img = img.astype(np.uint8)

    centroid = np.asarray(np.flip(centroids[idx_max],0))

    # Discard everything if the area is not big enough
    threshold_area = 7000
    validation = True
    if max_area < threshold_area:
        validation = False

    return img, centroid, validation

def PCA_direction(segmented_CC, centroid):
    # https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
    y, x = np.nonzero(segmented_CC)
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    evec1, evec2 = evecs[:, sort_indices]
    x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evec2
    if y_v1 > 0:        # staehlii: cheat to avoid orientation jumping around
        y_v1 *= -1
    scale = 100
    x1 = int(x_v1*-scale + centroid[1])
    x2 = int(x_v1*scale + centroid[1])
    y1 = int(y_v1*-scale + centroid[0])
    y2 = int(y_v1*scale + centroid[0])
    return x1, y1, x2, y2

######################################################################################
# Main Program

config = configparser.ConfigParser()
config.read('../../cfg/cfg.txt')

path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')
path_gestures = '../../../../Polybox/Shared/stove-state-data/ssds/gestures/'
video_format = '.mp4'
# video_format = '.h264'

path_labels = ''
file_name = 'I_2017-04-06-20_08_45_begg'
file_name = 'I_2017-04-13-21_26_55_begg'

path_video = join(path_videos, file_name + video_format)
path_gesture = join(path_gestures, 'begg', '1', file_name + '.h264')
cap = cv2.VideoCapture(path_gesture)
cap_video = cv2.VideoCapture(path_video)
ret, background = cap_video.read()
background = cv2.resize(background, (0, 0), fx=0.5, fy=0.5)


centroid_old = np.array([0,0])
centroid_vel = np.array([0,0],dtype=np.float)
features_file = open(join(path_labels, file_name + "_features.csv"), "w")
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == False:
        break
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    dim = frame.shape

    # Color Segmentation --------------------------------------------
    segmented = segmentation_RGB(frame)
    background_segmented = segmentation_RGB(background)
    segmented_sub = segmented - background_segmented

    # Connected Components -----------------------------------------
    segmented_final, centroid, validation = connected_components(segmented)
    if not validation:
        segmented_final = segmented_final*0

    segmented_final_color = cv2.cvtColor(segmented_final, cv2.COLOR_GRAY2RGB)
    if validation:
        # Compute centroid velocity ---------------------------------------------
        centroid_vel = centroid - centroid_old
        centroid_old = centroid
        vel_abs = math.sqrt(centroid_vel[0]**2 + centroid_vel[1]**2)

        # Compute Hand Orientation using PCA -------------------------------------
        a, b, c, d = PCA_direction(segmented_final, centroid)
        orientation = math.atan((b - d) / (a - c))

        # Plot centroid and orientation ------------------------------------------
        frame = cv2.circle(frame, (int(centroid[1]), int(centroid[0])), 10, (0, 255, 0), -1)
        segmented_final_color = cv2.circle(segmented_final_color, (int(centroid[1]), int(centroid[0])), 10, (0, 255, 0), -1)
        frame = cv2.line(frame, (a, b), (c, d), (0, 0, 255), 10)
        segmented_final_color = cv2.line(segmented_final_color, (a, b), (c, d), (0, 0, 255), 10)

        # Write features --------------------------------------------------------
        print("ctr_u, ctr_v, vel_u, vel_v, orient: {} {} {}".format(centroid, centroid_vel, orientation*180/3.1415))
        features_file.write(str(centroid[0]) + " " + str(centroid[1]) + " " + str(centroid_vel[0]) + " " + str(
            centroid_vel[1]) + " " + str(orientation) + "\n")



    # Display Images
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    cv2.resizeWindow("Frame", int(dim[1]/2), int(dim[0]/2))
    cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)
    cv2.imshow("Segmented", segmented_final_color)  #segmented_final_color
    cv2.resizeWindow("Segmented", int(dim[1] / 2), int(dim[0] / 2))
    k = cv2.waitKey(1)
    if k == 27:  # Exit by pressing escape-key
       break

features_file.close()
cv2.destroyAllWindows()
#####################################################################################