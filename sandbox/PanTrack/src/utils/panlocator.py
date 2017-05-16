import cv2
import random
import scipy
from matplotlib import pyplot as plt
from math import cos, sin, pi, inf, sqrt
from fitEllipse import *

from helpers import histogram_equalization, points_to_line, get_max_clique


class PanLocator:

    def __init__(self, _ellipse_smoothing='VOTE', _ellipse_method='CONVEX', _plot_ellipse=True):
        self._ellipse_smoothing = _ellipse_smoothing
        self._ellipse_method = _ellipse_method
        self._plot_ellipse = _plot_ellipse
        self.ellipse_counter = 0
        
        # Vote Parameters and Containers
        self.res_center = 300
        self.res_phi = 180
        self.res_center = 300
        self.accu_center = np.zeros((2, self.res_center))
        self.accu_phi = np.zeros((1, self.res_phi))
        self.accu_axes = np.zeros((2, self.res_center))

        # Best Candidate Parameters
        self.center = []
        self.axes = []
        self.phi = 0

    def reset_voting(self):
        self.ellipse_counter = 0
        self.accu_center = np.zeros((2, self.res_center))
        self.accu_phi = np.zeros((1, self.res_phi))
        self.accu_axes = np.zeros((2, self.res_center))

    def find_pan(self, patch, _plot_ellipse=False):

        self.ellipse_counter += 1

        raw_center, raw_axes, raw_phi, x, y = self.locate_pan(patch, _plot_ellipse=_plot_ellipse, method=self._ellipse_method)
        raw_center = raw_center[::-1]
        raw_axes = raw_axes[::-1]

        if self._ellipse_smoothing == 'AVERAGE':
            if self.ellipse_counter == 1:
                self.center, self.axes, self.phi = raw_center, raw_axes, raw_phi
            else:
                self.center = (self.center * (self.ellipse_counter - 1) + raw_center) / self.ellipse_counter
                self.axes = (self.axes * (self.ellipse_counter - 1) + raw_axes) / self.ellipse_counter
                self.phi = (self.phi * (self.ellipse_counter - 1) + raw_phi) / self.ellipse_counter
        elif self._ellipse_smoothing == 'VOTE':

            patch_size = patch.shape
            self.accu_center[0, int(raw_center[0] / patch_size[0] * self.res_center)] += 1
            self.accu_center[1, int(raw_center[1] / patch_size[1] * self.res_center)] += 1
            self.accu_axes[0, np.min([self.res_center - 1, int(raw_axes[0] / (patch_size[0]) * self.res_center)])] += 1
            self.accu_axes[1, np.min([self.res_center - 1, int(raw_axes[1] / (patch_size[1]) * self.res_center)])] += 1
            self.accu_phi[0, int(raw_phi/pi*self.res_phi)] += 1

            if self.ellipse_counter < 3:
                self.center, self.axes, self.phi = raw_center, raw_axes, raw_phi
            else:
                self.center[0] = np.argmax(self.accu_center[0, :])*patch_size[0]/self.res_center
                self.center[1] = np.argmax(self.accu_center[1, :])*patch_size[1]/self.res_center
                self.axes[0] = np.argmax(self.accu_axes[0, :])*(patch_size[0])/self.res_center
                self.axes[1] = np.argmax(self.accu_axes[1, :])*(patch_size[1])/self.res_center
                self.phi = np.argmax(self.accu_phi)*pi/self.res_phi

        elif self._ellipse_smoothing == 'RAW':
            self.center, self.axes, self.phi = raw_center, raw_axes, raw_phi

        return self.center, self.axes, self.phi

    def locate_pan(self, img, rgb=False, histeq=True, _plot_canny=False, _plot_cnt=False, _plot_ellipse=False, method='MAX_ARC'):
        plt.ion()

        if histeq:
            img = histogram_equalization(img)
        if rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        canny = cv2.Canny(img, 150, 300)

        if _plot_canny:
            plt.subplot(211), plt.imshow(img, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(212), plt.imshow(canny, cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            plt.show()


        # Find contours
        im2, contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)[:10]

        edges = []

        if _plot_ellipse:
            plt.subplot(211), plt.imshow(canny, cmap='gray')
            plt.title('Original Canny'), plt.xticks([]), plt.yticks([])
            plt.subplot(212), plt.imshow(canny, cmap='gray')
            plt.title('Elipses'), plt.xticks([]), plt.yticks([])
            plt.imshow(img, cmap='gray', zorder=1)

        if method == 'RANSAC':
            nr_samples = 3
            nr_iterations = 40
            max_score = 0

            for it in range(nr_iterations):
                edges = []

                candidate_cnts = random.sample(contours, nr_samples)
                mask = np.zeros(canny.shape, np.uint8)

                for cnt in candidate_cnts:
                    # mask contours
                    cv2.drawContours(mask, [cnt], 0, 255, -1)

                edge = mask * canny

                # Get individual pixels from mask
                pixelpoints = np.transpose(np.nonzero(edge))
                x = pixelpoints[:, 0]
                y = pixelpoints[:, 1]

                a = fitEllipse(x, y)
                center = ellipse_center(a)
                # phi = ellipse_angle_of_rotation(a)
                phi = ellipse_angle_of_rotation2(a)
                axes = ellipse_axis_length(a)

                arc_ = 2
                R_ = np.arange(0, arc_ * np.pi, 0.01)
                a, b = axes
                xx = center[0] + a * np.cos(R_) * np.cos(phi) - b * np.sin(R_) * np.sin(phi)
                yy = center[1] + a * np.cos(R_) * np.sin(phi) + b * np.sin(R_) * np.cos(phi)

                curr_score = 0
                # todo: pad mask

                for (x_it, y_it) in zip(xx, yy):
                    if mask.shape[0] > x_it > 0 and mask.shape[1] > y_it > 0:
                        if mask[int(y_it), int(x_it)] == 255:
                            curr_score += 1

                if curr_score*(axes[0] + axes[1]) > max_score:
                    if phi > 3.1415/2:
                        phi = phi - 3.1415/2

                    x_max = x
                    y_max = y
                    phi_max = phi
                    axes_max = axes
                    center_max = center
                    max_score = curr_score*(axes[0] + axes[1])
                    #print('NEW MAX SCORE: {}'.format(max_score))

        elif method == 'MAX_ARC':
            max_axes = 0

            for cnt in contours:
                # mask contours
                mask = np.zeros(canny.shape, np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)

                edge = mask * canny
                edges.append(edge)

                if _plot_cnt:
                    plt.subplot(211), plt.imshow(mask, cmap='gray')
                    plt.title('Mask'), plt.xticks([]), plt.yticks([])
                    plt.subplot(212), plt.imshow(edge, cmap='gray')
                    plt.title('Contour'), plt.xticks([]), plt.yticks([])
                    plt.show()

                pixelpoints = np.transpose(np.nonzero(edge))
                x = pixelpoints[:, 0]
                y = pixelpoints[:, 1]

                a = fitEllipse(x, y)
                center = ellipse_center(a)
                # phi = ellipse_angle_of_rotation(a)
                phi = ellipse_angle_of_rotation2(a)
                axes = ellipse_axis_length(a)

                if max_axes < axes[0] + axes[1]:
                    max_axes = axes[0] + axes[1]

                    if phi > 3.1415/2:
                        phi = phi - 3.1415/2

                    arc_ = 2
                    R_ = np.arange(0, arc_ * np.pi, 0.01)
                    a, b = axes
                    xx = center[0] + a * np.cos(R_) * np.cos(phi) - b * np.sin(R_) * np.sin(phi)
                    yy = center[1] + a * np.cos(R_) * np.sin(phi) + b * np.sin(R_) * np.cos(phi)
                    x_max = x
                    y_max = y
                    phi_max = phi
                    axes_max = axes
                    center_max = center

        elif method == 'CONVEX':

            max_axes = 0

            # Parameters
            tangent_range = 20      # How many pixels are to be considered for tangent fitting into both directions
            tangent_length = 100    # How long the plotted tangent is
            tangent_ratio_threshold = 0.2   # Edges must have tangent_ratio lower than this one to be considered convex
            convex_ratio_threshold = 0.1    # 1/threshold times more points need to be on the right side of the tangent
            angle_resolution = 180  # Number of angles to be checked for tangent between 1° and 180°
            _plot_tangent = False
            _plot_convex_edges = False
            _plot_max_clique = False
            convex_edges = []
            convex_tangents = []

            # for cnt in contours:
            #     mask = np.zeros(canny.shape, np.uint8)
            #     cv2.drawContours(mask, [cnt], 0, 255, -1)
            #     edge = mask * canny
            #     pixelpoints = np.transpose(np.nonzero(edge))
            #
            #     line = scipy.interpolate.Akima1DInterpolator(pixelpoints[:, 0], pixelpoints[:, 1])
            #     for point in pixelpoints:
            #         print(line(point[1]) - point[0])

            for cnt in contours:
                mask = np.zeros(canny.shape, np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)

                edge = mask * canny

                pixelpoints = np.transpose(np.nonzero(edge))

                center_point_id = int(len(pixelpoints)/2)

                # find center point
                center_point = pixelpoints[center_point_id, :]
                # draw tangent
                min_dis = inf
                best_r = 0
                best_theta = 0
                for theta in np.linspace(1, 180, angle_resolution):
                    # compute line
                    x_cent = center_point[1]
                    y_cent = center_point[0]

                    # https://en.wikipedia.org/wiki/Hough_transform
                    # http://www.intmath.com/plane-analytic-geometry/perpendicular-distance-point-line.php
                    r = x_cent*cos(theta*pi/180) + y_cent*sin(theta*pi/180)
                    sum_dis = 0
                    # todo replace with get_adjacent_points()
                    # todo find more points if many are discarded.
                    for point in pixelpoints[(center_point_id - tangent_range):(center_point_id + tangent_range), :]:
                        dis = abs(point[1]*cos(theta*pi/180) + point[0]*sin(theta*pi/180) - r) / sqrt(cos(theta*pi/180)**2 + sin(theta*pi/180)**2)
                        if sqrt((point[1] - x_cent)**2 + (point[0] - y_cent)**2) < tangent_range:
                            sum_dis += dis

                    # print('distance: {} angle: {}'.format(sum_dis, theta))

                    if sum_dis < min_dis:
                        min_dis = sum_dis
                        best_r = r
                        best_theta = theta

                tangent_start_x = x_cent + sin(best_theta*pi/180)*tangent_length
                tangent_start_y = y_cent - cos(best_theta*pi/180)*tangent_length
                tangent_end_x = x_cent - sin(best_theta*pi/180)*tangent_length
                tangent_end_y = y_cent + cos(best_theta*pi/180)*tangent_length

                if _plot_tangent:
                    plt.subplot(211), plt.imshow(mask, cmap='gray')
                    plt.title('Mask'), plt.xticks([]), plt.yticks([])
                    plt.subplot(212), plt.imshow(edge, cmap='gray')
                    plt.title('Contour'), plt.xticks([]), plt.yticks([])
                    plt.subplot(212), plt.scatter(x_cent, y_cent, color='red', s=5, zorder=3)
                    plt.plot([tangent_start_x, tangent_end_x], [tangent_start_y, tangent_end_y], color='red', linestyle='-', linewidth=1, zorder=20)
                    # for point in pixelpoints[(center_point_id - tangent_range):(center_point_id + tangent_range), :]:
                    #     if sqrt((point[1] - x_cent)**2 + (point[0] - y_cent)**2) < tangent_range:
                    #         plt.scatter(point[1], point[0], color='green', s=5, zorder=4)

                tangent_ratio, tangent_direction = points_to_line(pixelpoints, best_theta, best_r, _plot_tangent=False)

                if tangent_ratio < tangent_ratio_threshold:
                    convex_edges.append(np.transpose(np.nonzero(edge)))
                    convex_tangents.append([best_theta, best_r, tangent_direction])

                plt.show()

            c = np.eye(len(convex_edges))

            if _plot_convex_edges:
                # convex_edges are points
                plt.figure()
                plt.imshow(img)
                for convex_edge in convex_edges:
                    plt.scatter(convex_edge[:, 1], convex_edge[:, 0])
                    plt.pause(1)

            # i: original
            # j: comparison
            convex_edges_i = convex_edges[:-1]
            convex_tangents_i = convex_tangents[:-1]

            convex_edges_j = convex_edges[1::]
            convex_tangents_j = convex_tangents[1::]

            for i, (convex_edge_i, convex_tangent_i) in enumerate(zip(convex_edges_i, convex_tangents_i)):
                # print('outer:{}'.format(i))

                for j, (convex_edge_j, convex_tangent_j) in enumerate(zip(convex_edges_j, convex_tangents_j)):
                    # print('inner:{}'.format(i))
                    ratio_ij, direction_ij = points_to_line(convex_edge_j, convex_tangent_i[0], convex_tangent_i[1], _plot_tangent=False)
                    ratio_ji, direction_ji = points_to_line(convex_edge_i, convex_tangent_j[0], convex_tangent_j[1], _plot_tangent=False)
                    # print(ratio_ij, direction_ij, convex_tangent_i[2])
                    # print(ratio_ji, direction_ji, convex_tangent_j[2])

                    if (ratio_ij < convex_ratio_threshold and direction_ij == convex_tangent_i[2]) and\
                       (ratio_ji < convex_ratio_threshold and direction_ji == convex_tangent_j[2]):
                        c[i, i+j+1] = 1

                convex_edges_j = convex_edges_j[1::]
                convex_tangents_j = convex_tangents_j[1::]

            found_cliques = get_max_clique(c)
            max_clique_len = 0
            max_cliques = []

            for clique in found_cliques:
                if len(clique) > max_clique_len:
                    max_clique_len = len(clique)
                    max_cliques = []
                    max_cliques.append(clique)
                elif len(clique) == max_clique_len:
                    max_cliques.append(clique)

            for clique in max_cliques:
                for it, edge_id in enumerate(clique):
                    if it == 0:
                        pixelpoints = convex_edges[edge_id]
                    else:
                        pixelpoints = np.append(pixelpoints, convex_edges[edge_id], axis=0)

                x = pixelpoints[:, 0]
                y = pixelpoints[:, 1]

                # Fit Ellipse and get parameters
                a = fitEllipse(x, y)
                center = ellipse_center(a)
                # phi = ellipse_angle_of_rotation(a)
                phi = ellipse_angle_of_rotation2(a)
                axes = ellipse_axis_length(a)

                # todo define better criterion than ellipse size
                if max_axes < axes[0] + axes[1]:
                    max_axes = axes[0] + axes[1]

                    if phi > 3.1415 / 2:
                        phi = phi - 3.1415 / 2

                    arc_ = 2
                    R_ = np.arange(0, arc_ * np.pi, 0.01)
                    a, b = axes
                    xx = center[0] + a * np.cos(R_) * np.cos(phi) - b * np.sin(R_) * np.sin(phi)
                    yy = center[1] + a * np.cos(R_) * np.sin(phi) + b * np.sin(R_) * np.cos(phi)
                    x_max = x
                    y_max = y
                    phi_max = phi
                    axes_max = axes
                    center_max = center
                    max_clique = pixelpoints

                if _plot_max_clique:
                    # convex_edges are points
                    plt.figure()
                    plt.imshow(img)
                    plt.scatter(pixelpoints[:, 1], pixelpoints[:, 0])
                    plt.pause(1)

        if _plot_ellipse:
            # plt.scatter(y, x,color='green', s=1, zorder=2)
            plt.scatter(yy, xx, color='red', s=1, zorder=3)
            # print('Phi ={}'.format(phi_max*180/3.1415))

        plt.show()

        return center_max,axes_max, phi_max, x_max, y_max