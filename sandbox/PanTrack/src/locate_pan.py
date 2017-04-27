import cv2
import random
from matplotlib import pyplot as plt
from math import cos, sin, pi, inf, sqrt
from fitEllipse import *


def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()

    # cdf_normalized = cdf * hist.max() / cdf.max()
    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(img.flatten(),256,[0,256], color = 'r')
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')
    # plt.show()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    return cdf[img]


def locate_pan(img, rgb=False, histeq=True, _plot_canny=False, _plot_cnt=False, _plot_ellipse=False, method='MAX_ARCH'):

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
            tangent_range = 10
            for theta in range(180):
                # compute line
                x_cent = center_point[1]
                y_cent = center_point[0]

                r = x_cent*cos(theta*pi/180) + y_cent*sin(theta*pi/180)
                dis = 0
                for point in pixelpoints[(center_point_id - tangent_range):(center_point_id + tangent_range), :]:
                    dis += abs(point[1]*cos(theta*pi/180) + point[0]*sin(theta*pi/180) - r) / sqrt(cos(theta*pi/180)**2 + sin(theta*pi/180)**2)

                print('distance: {} angle: {}'.format(dis, theta))

                if dis < min_dis:
                    min_dis = dis
                    best_r = r
                    best_theta = theta

            print(best_theta)

            tangent_length = 20

            tangent_start_x = x_cent + sin(best_theta*pi/180)*tangent_length
            tangent_end_x = x_cent - sin(best_theta*pi/180)*tangent_length

            tangent_start_y = y_cent + cos(best_theta*pi/180)*tangent_length
            tangent_end_y = y_cent - cos(best_theta*pi/180)*tangent_length

            print(tangent_end_y)
            print(tangent_start_y)

            plt.subplot(211), plt.imshow(mask, cmap='gray')
            plt.title('Mask'), plt.xticks([]), plt.yticks([])
            plt.subplot(212), plt.imshow(edge, cmap='gray')
            plt.title('Contour'), plt.xticks([]), plt.yticks([])
            plt.subplot(212), plt.scatter(center_point[1], center_point[0], color='red', s=5, zorder=3)
            plt.plot([tangent_start_x, tangent_end_x], [tangent_start_y, tangent_end_y], color='red', linestyle='-', linewidth=2)
            plt.show()

            # Group Convex contours

        if _plot_ellipse:
            # plt.scatter(y, x,color='green', s=1, zorder=2)
            plt.scatter(yy, xx, color='red', s=1, zorder=3)
            # print('Phi ={}'.format(phi_max*180/3.1415))
        plt.show()

    return center_max,axes_max, phi_max, xx, yy