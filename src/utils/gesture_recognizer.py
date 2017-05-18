from utils_gesture import *
import cv2
import math
import configparser
import pickle

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
            N = 12
            self.features = np.array(self.features)
            STF = extract_STF(self.features, N)

            config = configparser.ConfigParser()
            config.read('../cfg/cfg.txt')
            path_videos = config.get('paths', 'videos')
            path_models = path_videos[0:-7] + '/gestures/models/'

            label_names = ['place pan', 'place food', 'place lid', 'remove lid', 'remove food', 'remove pan', 'season',
                           'other']
            predictions = np.zeros((len(label_names), 1), dtype=np.int)
            for i in range(0,N):
                clf = pickle.load(open(path_models + 'model_STF_{}'.format(i), 'rb'))
                predicted_label = clf.predict(STF[:,i])
                predictions[predicted_label - 1] += 1

            best_label_idx = predictions.argmax()
            gesture = label_names[best_label_idx]
            self.features = []

        self.old_hand_in_frame = all(self.hand_in_frame)

        cv2.namedWindow("Segmentation", 2)
        # cv2.resizeWindow("Segmentation", 640, 480)
        cv2.imshow("Segmentation", segmented_final)


        return gesture
