import cv2
import configparser
import numpy as np
import os
import pickle

from math import pi

# Own Libraries
from panlocator import PanLocator
from helpers import *


class Recognition:

    def __init__(self):
        # Params
        self._ellipse_smoothing = 'AVERAGE'
        self._ellipse_smoothing = 'RAW'
        self._ellipse_smoothing = 'VOTE'

        self._ellipse_method = 'RANSAC'
        self._ellipse_method = 'MAX_ARC'
        self._ellipse_method = 'CONVEX'

        self._segment = False

        # Read config
        self.cfg_path = '../cfg/class_cfg.txt'
        self.config = configparser.ConfigParser()
        self.config.read(self.cfg_path)

        self.polybox_path = self.config.get('paths', 'polybox')

        self.pan_models_path = self.polybox_path + 'pan_detect/pan_models/'
        self.food_models_path = self.polybox_path + 'pan_detect/food_models/'

        self.pan_model_name = '2017-05-11-16_44_38'
        self.pan_model_name = '2017-04-27-15_19_51'  # I_begg1
        self.pan_model_name = '2017-05-17-23_54_57'  # I_2 segg/scegg

        self.food_model_name = '2017-05-18-14_19_44'

        # Load pan detect model
        self.pan_model = pickle.load(open(self.pan_models_path + 'M_' + self.pan_model_name + '.sav', 'rb'))
        self.food_model = pickle.load(open(self.food_models_path + 'M_' + self.food_model_name + '.sav', 'rb'))

        # Load pan_model info file
        with open(self.pan_models_path + 'I_' + self.pan_model_name + '.txt', 'r') as file:
            self._pan_params = eval(file.read())

        print('Pan model parameters: ')
        for key, val in self._pan_params.items():
            print('\t{}: {}'.format(key, val))

        # Load food_model info file
        with open(self.food_models_path + 'I_' + self.food_model_name + '.txt', 'r') as file:
            self._food_params = eval(file.read())

        print('Food model parameters: ')
        for key, val in self._food_params.items():
            print('\t{}: {}'.format(key, val))

        # Read corners and reshape them into 2d-Array
        self.corners = np.reshape(self._pan_params['corners'], (-1, 4))
        self.plate_of_interest = int(self._pan_params['plate_of_interest'])

        # import images or videos or video stream
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

        # init pan_locator
        self.pan_locator = PanLocator(_ellipse_smoothing=self._ellipse_smoothing, _ellipse_method=self._ellipse_method)

        self.center = []
        self.axes = []
        self.phi = []

    def process_frame(self, frame):

        food_label_predicted_name = []

        patch = frame[self.corners[self.plate_of_interest - 1, 1]:self.corners[self.plate_of_interest - 1, 3],
                      self.corners[self.plate_of_interest - 1, 0]:self.corners[self.plate_of_interest - 1, 2]]

        patch_normalized = histogram_equalization(patch)

        pan_feature = get_HOG(patch_normalized,
                              orientations=self._pan_params['feature_params']['orientations'],
                              pixels_per_cell=self._pan_params['feature_params']['pixels_per_cell'],
                              cells_per_block=self._pan_params['feature_params']['cells_per_block'],
                              widthPadding=self._pan_params['feature_params']['widthPadding'])

        pan_label_predicted_id = self.pan_model.predict(pan_feature)
        pan_label_predicted_name = self._pan_params['labels'][int(pan_label_predicted_id)]

        if 'pan' in pan_label_predicted_name or 'lid' in pan_label_predicted_name:

            self.center, self.axes, self.phi = self.pan_locator.find_pan(patch)

            # Run Object Recognition inside pan
            mask = np.zeros_like(patch)
            ellipse_mask = cv2.ellipse(mask, tuple(map(int, self.center)), tuple(map(int, self.axes)),
                                       int(-self.phi * 180 / pi), 0, 360, (255, 255, 255), thickness=-1)

            if self._food_params['feature_type'] == 'RGB_HIST':
                food_feature = np.zeros((3, self._food_params['feature_params']['resolution']))

                for i in range(3):
                    food_feature[i, :] = np.transpose(
                        cv2.calcHist([patch], [i], ellipse_mask[:, :, 0],
                                     [self._food_params['feature_params']['resolution']], [0, 256]))

                food_feature = food_feature.flatten()

            food_label_predicted_id = self.food_model.predict(food_feature)
            food_label_predicted_name = self._food_params['labels'][int(food_label_predicted_id)]

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

        cv2.putText(plot_patch, str(pan_label_predicted_name), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0))
        cv2.putText(plot_patch, str(food_label_predicted_name), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0))
        cv2.imshow('predicted', plot_patch)
        cv2.waitKey(1)

        return pan_label_predicted_name, food_label_predicted_name

    def get_pan_location(self):
        return self.center, self.axes, self.phi