import cv2
import random
from matplotlib import pyplot as plt
from math import cos, sin, pi, inf, sqrt
from fitEllipse import *

from helpers import histogram_equalization, points_to_line, get_max_clique


def locate_pan(img, rgb=False, histeq=True, _plot_canny=False, _plot_cnt=False, _plot_ellipse=False, method='MAX_ARCH'):
    plt.ion()

    if histeq:
        img = histogram_equalization(img)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(img, 100, 200)

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

    max_axes = 0

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

    elif method == 'MAX_ARCH':
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

        # Parameters
        tangent_range = 20      # How many pixels are to be considered for tangent fitting into both directions
        tangent_length = 100    # How long the plotted tangent is
        tangent_ratio_threshold = 0.2   # Edges must have tangent_ratio lower than this one to be considered convex
        convex_ratio_threshold = 0.1    # 1/threshold times more points need to be on the right side of the tangent
        angle_resolution = 180  # Number of angles to be checked for tangent between 1° and 180°
        _plot_tangent = False
        _plot_convex_edges = False
        convex_edges = []
        convex_tangents = []

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

        # convex_edges are points
        convex_comp_edges = convex_edges[1::]
        convex_comp_tangents = convex_tangents[1::]

        convex_edges = convex_edges[:-1]
        convex_tangents = convex_tangents[:-1]


        plt.figure()
        # todo Group convex Contours
        plt.imshow(img)
        for i, (convex_edge, convex_tangent) in enumerate(zip(convex_edges, convex_tangents)):
            plt.scatter(convex_edge[:,1], convex_edge[:,0])
            plt.pause(1)
            for j, (convex_comp_edge, convex_comp_tangent) in enumerate(zip(convex_comp_edges, convex_comp_tangents)):
                ratio_ij, direction_ij = points_to_line(convex_comp_edge, convex_tangent[0], convex_tangent[1], _plot_tangent=False)
                ratio_ji, direction_ji = points_to_line(convex_edge, convex_comp_tangent[0], convex_comp_tangent[1], _plot_tangent=False)
                #print(ratio_ij, direction_ij, convex_tangent[2])
                #print(ratio_ji, direction_ji, convex_comp_tangent[2])

                if (ratio_ij < convex_ratio_threshold and direction_ij == convex_tangent[2]) and\
                   (ratio_ji < convex_ratio_threshold and direction_ji == convex_comp_tangent[2]):
                    c[i, i+j+1] = 1

                convex_comp_edges = convex_comp_edge[1::]
                convex_comp_tangents = convex_comp_tangents[1::]


        # print(c)

        max_clique_graph = get_max_clique(c)

        print(max_clique_graph.nodes())

        pixelpoints = np.transpose(np.nonzero(edge))
        x = pixelpoints[:, 0]
        y = pixelpoints[:, 1]

        a = fitEllipse(x, y)
        center = ellipse_center(a)
        # phi = ellipse_angle_of_rotation(a)
        phi = ellipse_angle_of_rotation2(a)
        axes = ellipse_axis_length(a)

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

    if _plot_ellipse:
        # plt.scatter(y, x,color='green', s=1, zorder=2)
        plt.scatter(yy, xx, color='red', s=1, zorder=3)
        # print('Phi ={}'.format(phi_max*180/3.1415))
    plt.show()

    return center_max,axes_max, phi_max, xx, yy