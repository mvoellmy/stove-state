import cv2
import configparser
import numpy as np
import os
import pickle

from math import pi

# Own Libraries
from panlocator import PanLocator
from helpers import *


class FoodRecognizer:

    def __init__(self, plate_of_interest, ellipse_smoothing='VOTE_SLIDE', ellipse_method='MAX_ARC'):
        # Params
        self._ellipse_smoothing = 'AVERAGE'
        self._ellipse_smoothing = 'RAW'
        self._ellipse_smoothing = 'VOTE'
        self._ellipse_smoothing = ellipse_smoothing

        self._ellipse_method = 'RANSAC'
        self._ellipse_method = 'CONVEX'
        self._ellipse_method = ellipse_method

        self._segment = False
        self._plate_of_interest = plate_of_interest

        # Read config
        self.cfg_path = '/Users/miro/Documents/Repositories/stove-state/cfg/class_cfg.txt'
        self.config = configparser.ConfigParser()
        self.config.read(self.cfg_path)

        self.polybox_path = self.config.get('paths', 'polybox')

        self.pan_models_path = self.polybox_path + 'pan_detect/pan_models/'
        self.food_models_path = self.polybox_path + 'pan_detect/food_models/'

        self.pan_model_name = '2017-05-11-16_44_38'

        if self._plate_of_interest == 'I_4':
            self.pan_model_name = '2017-05-18-18_25_11'   # I_4 begg1 hog
            self.food_model_name = '2017-05-19-09_20_36'  # I_4 poly rgb_hist
            self.food_model_name = '2017-06-10-18_23_58'  # I_4 First tf-idf test
            self.food_model_name = '2017-06-10-18_52_46'  # I_4 SIFT + tf-idf
            self.food_model_name = '2017-06-04-17_40_02'  # I_4 SIFT
            self.food_model_name = '2017-06-11-18_44_11'  # I_4 SIFT + tf-idf 2

        elif self._plate_of_interest == 'I_2':
            self.pan_model_name = '2017-05-18-17_03_24'   # I_2 segg/scegg hog
            self.food_model_name = '2017-05-18-14_19_44'  # rgb_hist

        else:
            print('ERROR: Invalid Plate of interest')

        # Load pan detect model
        self.pan_model = pickle.load(open(self.pan_models_path + 'M_' + self.pan_model_name + '.sav', 'rb'))
        self.food_model = pickle.load(open(self.food_models_path + 'M_' + self.food_model_name + '.sav', 'rb'))

        # Load pan_model info file
        with open(self.pan_models_path + 'I_' + self.pan_model_name + '.txt', 'r') as file:
            self._pan_params = eval(file.read())

        print('Pan model parameters: ')
        for key, val in self._pan_params.items():
            if str(key) != 'visual_word_idf':
                print('\t{}: {}'.format(key, val))

        # Load food_model info file
        with open(self.food_models_path + 'I_' + self.food_model_name + '.txt', 'r') as file:
            self._food_params = eval(file.read())

        print('Food model parameters: ')
        for key, val in self._food_params.items():
            if str(key) != 'visual_word_idf':
                print('\t{}: {}'.format(key, val))

        # Read corners and reshape them into 2d-Array
        self.corners = np.reshape(self._pan_params['corners'], (-1, 4))
        self.plate_of_interest = int(self._pan_params['plate_of_interest'])

        if self._food_params['feature_type'] == 'SIFT':
            self.kmeans = pickle.load(open(self.food_models_path + 'K_' + self.food_model_name + '.sav', 'rb'))


        # import images or videos or video stream
        # self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

        # init pan_locator
        self.pan_locator = PanLocator(_ellipse_smoothing=self._ellipse_smoothing, _ellipse_method=self._ellipse_method)

        self.center = []
        self.global_center = []
        self.axes = []
        self.phi = []

        self.food_label_predicted_name = []
        self.food_label_predicted_id = []
        self.pan_label_predicted_name = []
        self.pan_label_predicted_id = []

        self.dead_feature_count = 0

    def process_frame(self, frame):

        food_label_predicted_name = []

        patch = np.copy(frame[self.corners[self.plate_of_interest - 1, 1]:self.corners[self.plate_of_interest - 1, 3],
                      self.corners[self.plate_of_interest - 1, 0]:self.corners[self.plate_of_interest - 1, 2]])

        patch_normalized = histogram_equalization(patch)

        pan_feature = get_HOG(patch_normalized,
                              orientations=self._pan_params['feature_params']['orientations'],
                              pixels_per_cell=self._pan_params['feature_params']['pixels_per_cell'],
                              cells_per_block=self._pan_params['feature_params']['cells_per_block'],
                              widthPadding=self._pan_params['feature_params']['widthPadding'])

        self.pan_label_predicted_id = self.pan_model.predict(pan_feature.reshape(1, -1))
        self.pan_label_predicted_name = self._pan_params['labels'][int(self.pan_label_predicted_id)]

        # if 'pan' in self.pan_label_predicted_name:
        if True:

            self.center, self.axes, self.phi = self.pan_locator.find_pan(patch)

            # Run Object Recognition inside pan
            mask = np.zeros_like(patch)
            ellipse_mask = cv2.ellipse(mask, tuple(map(int, self.center)), tuple(map(int, self.axes)),
                                       int(-self.phi * 180 / pi), 0, 360, (255, 255, 255), thickness=-1)

            if len(ellipse_mask.shape) > 2:
                ellipse_mask = ellipse_mask[:, :, 0]

            if self._food_params['feature_type'] == 'RGB_HIST':
                food_feature = np.zeros((3, self._food_params['feature_params']['resolution']))

                for i in range(3):
                    food_feature[i, :] = np.transpose(
                        cv2.calcHist([patch], [i], ellipse_mask,
                                     [self._food_params['feature_params']['resolution']], [0, 256]))

                food_feature = food_feature.flatten()

            if self._food_params['feature_type'] == 'SIFT':
                sift = cv2.xfeatures2d.SIFT_create()
                kp, descriptors = sift.detectAndCompute(patch, ellipse_mask)

                food_feature = np.zeros(self._food_params['feature_params']['k'])
                # Create
                if descriptors is None:
                    self.dead_feature_count += 1
                else:
                    sift_stack = np.reshape(descriptors, (-1, 128))

                    for sift_feature in sift_stack:
                        food_feature[self.kmeans.predict(sift_feature.reshape(1, -1))] += 1

                    if self._food_params['feature_params']['tf-idf']:
                        tf = np.zeros(self._food_params['feature_params']['k'])
                        idf = self._food_params['visual_word_idf']

                        for i, visual_word_count in enumerate(food_feature):
                            tf[i] = visual_word_count/sum(food_feature)

                        food_feature = food_feature * tf * idf

            self.food_label_predicted_id = self.food_model.predict(food_feature.reshape(1, -1))
            self.food_label_predicted_name = self._food_params['labels'][int(self.food_label_predicted_id)]

            # Plot contures of used edges
            # for x_it, y_it in zip(x, y):
            #    cv2.circle(patch, (y_it, x_it), 2, (0, 255, 0), -1)

            # Plot ellipse of fitted ellipse
            # cv2.ellipse(patch, tuple(map(int, raw_center)), tuple(map(int, raw_axes)),
            #             int(-raw_phi*180/pi), 0, 360, (255, 0, 0), thickness=2)

            # Plot ellipse of voted ellipse
            cv2.ellipse(patch, tuple(map(int, self.center)), tuple(map(int, self.axes)),
                        int(-self.phi * 180 / pi), 0, 360, (0, 0, 255), thickness=5)

            # # masked_patch = masked_patch[]
            # if _segment:
            #     masked_patch = np.bitwise_and(patch, ellipse_mask)
            #     fgmask = fgbg.apply(patch)
            #     fgmask = np.dstack((fgmask, fgmask, fgmask))
            #     plot_patch = np.bitwise_and(masked_patch, fgmask)
            #     plot_patch = masked_patch
            # else:
            #     plot_patch = patch

            plot_patch = patch

        else:
            plot_patch = patch

        # cv2.putText(plot_patch, str(pan_label_predicted_name), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0))
        # cv2.putText(plot_patch, str(self.food_label_predicted_name), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0))
        # cv2.imshow('predicted', plot_patch)
        # cv2.waitKey(1)

        return self.pan_label_predicted_name, self.food_label_predicted_name, self.pan_label_predicted_id, self.food_label_predicted_id

    def get_pan_location(self):

        self.global_center = np.zeros((2,1))
        self.global_center[0] = self.center[0] + self.corners[self.plate_of_interest - 1, 0]
        self.global_center[1] = self.center[1] + self.corners[self.plate_of_interest - 1, 1]

        return self.global_center, self.axes, self.phi

    def reset_pan_location(self):
        self.pan_locator.reset_voting()

    def get_models(self):
        return self.pan_model_name,\
               self.pan_models_path,\
               self.food_model_name,\
               self.food_models_path

    def get_dead_features(self):
        return self.dead_feature_count
