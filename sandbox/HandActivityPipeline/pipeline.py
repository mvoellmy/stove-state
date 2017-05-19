import numpy as np
import cv2
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

def segmentation_YCC(frame):
    # http://stackoverflow.com/questions/14752006/computer-vision-masking-a-human-hand
    imgYCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    Y = imgYCC[:, :, 0]
    Cr = imgYCC[:, :, 1]
    Cb = imgYCC[:, :, 2]
    skin_ycrcb_mint = np.array((0, 150, 0))
    skin_ycrcb_maxt = np.array((255, 255, 255))
    skin_ycrcb = cv2.inRange(imgYCC, skin_ycrcb_mint, skin_ycrcb_maxt)

    return skin_ycrcb

def segmentation_HSV(frame):
    # http://stackoverflow.com/questions/14752006/computer-vision-masking-a-human-hand
    skin_min = np.array([0, 0, 120], np.uint8)
    skin_max = np.array([20, 255, 255], np.uint8)
    gaussian_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    blur_hsv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)
    segmented = cv2.inRange(blur_hsv, skin_min, skin_max)

    return segmented

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
    y_v2, x_v2 = evec2

    if abs(x_v1) > abs(y_v1) and np.sign(x_v1) == 1: # staehlii: cheat to avoid orientation jumping around
        y_v1 = - y_v1

    scale = 100
    x1 = int(x_v1*-scale + centroid[1])
    x2 = int(x_v1*scale + centroid[1])
    y1 = int(y_v1*-scale + centroid[0])
    y2 = int(y_v1*scale + centroid[0])
    return x1, y1, x2, y2

######################################################################################

def pipeline(cap, cap_video, path_feature_file=[], path_video_file=[]):
    # ret, background = cap_video.read()
    # background = cv2.resize(background, (0, 0), fx=0.5, fy=0.5)
    # background_segmented = segmentation_RGB(background)
    # background_final, centroid, validation = connected_components(background_segmented)

    centroid_old = [] #np.array([0,0])
    centroid_vel = np.array([0,0],dtype=np.float)
    if path_feature_file != []:
        print(path_feature_file)
        features_file = open(path_feature_file, "w")
    out = []
    hand_in_frame = []
    gesture_history = 0
    condition = False
    bMHI = []
    MH_length = 10
    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        dim = frame.shape

        # Color Segmentation --------------------------------------------
        segmented = segmentation_YCC(frame)
        # segmented_sub = segmented - background_final

        # Connected Components -----------------------------------------
        segmented_final, centroid, validation = connected_components(segmented)
        if not validation:
            segmented_final = segmented_final*0

        segmented_final_color = cv2.cvtColor(segmented_final, cv2.COLOR_GRAY2RGB)
        if validation:
            # Compute centroid velocity ---------------------------------------------
            if centroid_old != []:
                centroid_vel = centroid - centroid_old
            vel_abs = math.sqrt(centroid_vel[0]**2 + centroid_vel[1]**2)

            # Compute Hand Orientation using PCA -------------------------------------
            a, b, c, d = PCA_direction(segmented_final, centroid)
            orientation = math.atan2((b - d) , (a - c))

            # Plot centroid and orientation ------------------------------------------
            frame = cv2.circle(frame, (int(centroid[1]), int(centroid[0])), 10, (0, 255, 0), -1)
            segmented_final_color = cv2.circle(segmented_final_color, (int(centroid[1]), int(centroid[0])), 10, (0, 255, 0), -1)
            frame = cv2.line(frame, (a, b), (c, d), (0, 0, 255), 10)
            segmented_final_color = cv2.line(segmented_final_color, (a, b), (c, d), (0, 0, 255), 10)

            # Write features --------------------------------------------------------
            if path_feature_file != [] and centroid_old != []:
                # print("ctr_u, ctr_v, vel_u, vel_v, orient: {} {} {}".format(centroid, centroid_vel, orientation*180/3.1415))
                features_file.write(str(centroid[0]) + " " + str(centroid[1]) + " " + str(centroid_vel[0]) + " " + str(
                    centroid_vel[1]) + " " + str(orientation) + "\n")
            centroid_old = centroid
        # HMM ----------------------------------------------------------------------
        num_history = 5
        if validation:
            hand_in_frame.append(True)
        else:
            hand_in_frame.append(False)
        if len(hand_in_frame) > num_history:
            del hand_in_frame[0]
        # print(hand_in_frame)

        gestures = ['nothing', 'place pan', 'place egg', 'place lid', 'remove lid', 'remove egg', 'remove pan']
        if not any(hand_in_frame):
            gesture_num = 0
            condition = False
        elif condition != any(hand_in_frame):
            condition = any(hand_in_frame)
            gesture_history += 1
            gesture_num = gesture_history
        # frame = cv2.putText(frame, gestures[gesture_num], (50, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 5)

        # # Binary Motion History Image (bMHI)
        # if bMHI == []:
        #     bMHI = [segmented_final/255]
        # else:
        #     bMHI.append(segmented_final/255)
        #     if len(bMHI) > MH_length:
        #         bMHI.pop(0)
        #         final_bHMI = bMHI[-1]*(MH_length+1)
        #         for i in range(0,MH_length-1):
        #             final_bHMI += bMHI[i]*(i+1)
        #         final_bHMI *= 255/np.max(final_bHMI)
        #         final_bHMI = final_bHMI.astype(np.uint8)
        #
        #         cv2.namedWindow("bMHI", cv2.WINDOW_NORMAL)
        #         cv2.imshow("bMHI", final_bHMI)
        #         cv2.resizeWindow("bMHI", int(dim[1] / 2), int(dim[0] / 2))


        # Display Images -----------------------------------------------------------
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", segmented)
        cv2.resizeWindow("Frame", int(dim[1]/2), int(dim[0]/2))
        cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)
        cv2.imshow("Segmented", segmented_final_color)  #segmented_final_color
        cv2.resizeWindow("Segmented", int(dim[1] / 2), int(dim[0] / 2))
        k = cv2.waitKey(1)
        if k == 27:  # Exit by pressing escape-key
            break

        # Write video file ---------------------------------------------------------
        if path_video_file != []:
            if out == []:
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                out = cv2.VideoWriter(path_video_file, fourcc, 25.0, (dim[1], dim[0]), True)
            out.write(frame)

    # Clear all objects
    cv2.destroyAllWindows()
    if path_feature_file != []:
        features_file.close()
    if path_video_file != []:
        out.release()
#####################################################################################