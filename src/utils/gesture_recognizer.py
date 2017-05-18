from utils_gesture import *
import cv2
import math

class GestureRecognizer(object):
    features = []
    centroid_old = []
    centroid_vel = []
    hand_in_frame = []
    old_hand_in_frame = False

    def __init__(self):
        pass

    def process_frame(self, frame):
        # Color Segmentation --------------------------------------------
        segmented = segmentation_YCC(frame)

        # Connected Components -----------------------------------------
        segmented_final, centroid, validation = connected_components(segmented)
        if not validation:
            segmented_final = segmented_final*0

        if validation:
            # Compute centroid velocity ---------------------------------------------
            if self.centroid_old != []:
                self.centroid_vel = centroid - self.centroid_old
            # vel_abs = math.sqrt(centroid_vel[0] ** 2 + centroid_vel[1] ** 2)

            # Compute Hand Orientation using PCA -------------------------------------
            a, b, c, d = PCA_direction(segmented_final, centroid)
            orientation = math.atan2((b - d), (a - c))

            if self.centroid_vel != []:
                feature = [centroid[0], centroid[1], self.centroid_vel[0], self.centroid_vel[1], orientation]
                self.features.append(feature)

        self.centroid_old = centroid

        num_history = 30
        if validation:
            self.hand_in_frame.append(True)
        else:
            self.hand_in_frame.append(False)
        if len(self.hand_in_frame) > num_history:
            del self.hand_in_frame[0]

        # gesture = (np.array(self.hand_in_frame*1)).sum() > num_history/2
        gesture = []
        if self.old_hand_in_frame and not all(self.hand_in_frame):
            gesture = 'gesture'

        self.old_hand_in_frame = all(self.hand_in_frame)

        cv2.namedWindow("Segmentation", 2)
        # cv2.resizeWindow("Segmentation", 640, 480)
        cv2.imshow("Segmentation", segmented_final)


        return gesture
