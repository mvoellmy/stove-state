'''
Script to fit an ellipse to a set of points.
- The ellipse is represented by the two foci and the length of a
     line segment which is drawn from the foci to the
     point where the ellipse intersects the minor axis.

- Fitting algorithm from Yu, Kulkarni & Poor
'''

__author__ = 'Ed Tate'
__email__ = 'edtate<at>gmail-dot-com'
__website__ = 'exnumerus.blogspot.com'
__license__ = 'Creative Commons Attribute By - http://creativecommons.org/licenses/by/3.0/us/'''

####################################################
# create ellipse with random noise in points
from random import uniform, normalvariate
from math import pi, sin, cos, exp, pi, sqrt
from openopt import NLP
from numpy import *
from numpy import linalg as LA
import matplotlib.pylab as pp


def gen_ellipse_pts(a, foci1, foci2,
                    num_pts=200, angles=None,
                    x_noise=None, y_noise=None):
    '''
       Generate points for an ellipse given
          the foci, and
          the distance to the intersection of the minor axis and ellipse.

       Optionally,
          the number of points can be specified,
          the angles for the points wrt to the centroid of the ellipse, and
          a noise offset for each point in the x and y axis.
    '''
    c = (1 / 2.0) * LA.norm(foci1 - foci2)
    b = sqrt(a ** 2 - c ** 2)
    x1 = foci1[0]
    y1 = foci1[1]
    x2 = foci2[0]
    y2 = foci2[1]
    if angles is None:
        t = arange(0, 2 * pi, 2 * pi / float(num_pts))
    else:
        t = array(angles)

    ellipse_x = (x1 + x2) / 2 + (x2 - x1) / (2 * c) * a * cos(t) - (y2 - y1) / (2 * c) * b * sin(t)
    ellipse_y = (y1 + y2) / 2 + (y2 - y1) / (2 * c) * a * cos(t) + (x2 - x1) / (2 * c) * b * sin(t)
    try:
        # try adding noise to the ellipse points
        ellipse_x = ellipse_x + x_noise
        ellipse_y = ellipse_y + y_noise
    except TypeError:
        pass
    return (ellipse_x, ellipse_y)


####################################################################

# setup the reference ellipse

# define the foci locations
foci1_ref = array([2, -1])
foci2_ref = array([-2, 1])
# pick distance from foci to ellipse
a_ref = 2.5

# generate points for reference ellipse without noise
ref_ellipse_x, ref_ellipse_y = gen_ellipse_pts(a_ref, foci1_ref, foci2_ref)

# generate list of noisy samples on the ellipse
num_samples = 1000
angles = [uniform(-pi, pi) for i in range(0, num_samples)]
sigma = 0.2
x_noise = [normalvariate(0, sigma) for t in angles]
y_noise = [normalvariate(0, sigma) for t in angles]
x_list, y_list = gen_ellipse_pts(a_ref, foci1_ref, foci2_ref,
                                 angles=angles,
                                 x_noise=x_noise,
                                 y_noise=y_noise)

point_list = []
for x, y in zip(x_list, y_list):
    point_list.append(array([x, y]))

# draw the reference ellipse and the noisy samples
pp.figure()
pp.plot(x_list, y_list, '.b', alpha=0.5)
pp.plot(ref_ellipse_x, ref_ellipse_y, 'g', lw=2)
pp.plot(foci1_ref[0], foci1_ref[1], 'o')
pp.plot(foci2_ref[0], foci2_ref[1], 'o')


#####################################################

def initialize():
    '''
    Determine the initial value for the optimization problem.
    '''
    # find x mean
    x_mean = array(x_list).mean()
    # find y mean
    y_mean = array(y_list).mean()
    # find point farthest away from mean
    points = array(zip(x_list, y_list))
    center = array([x_mean, y_mean])
    distances = zeros((len(x_list), 1))
    for i, point in enumerate(points):
        distances[i, 0] = LA.norm(point - center)
    ind = where(distances == distances.max())
    max_pt = points[ind[0], :][0]
    # find point between mean and max point
    foci1 = (max_pt + center) / 2.0
    # find point opposite from
    foci2 = 2 * center - max_pt
    return [distances.max(), foci1[0], foci1[1], foci2[0], foci2[1]]


def objective(x):
    '''
    Calculate the objective cost in the optimization problem.
    '''
    foci1 = array([x[1], x[2]])
    foci2 = array([x[3], x[4]])
    a = x[0]
    n = float(len(point_list))
    _lambda = 0.1
    _sigma = sigma
    sum = 0
    for point in point_list:
        sum += ((LA.norm(point - foci1, 2) + LA.norm(point - foci2, 2) - 2 * a) ** 2) / n
    sum += _lambda * ahat_max * _sigma * exp((a / ahat_max) ** 4)
    return sum


# solve the optimization problem
x0 = initialize()
ahat_max = x0[0]
print
x0
p = NLP(objective, x0)
r = p.solve('ralg')
print
r.xf

# get the results from the optimization problem
xf = r.xf
# unload the specific values from the result vector
foci1 = array([xf[1], xf[2]])
foci2 = array([xf[3], xf[4]])
a = xf[0]

# reverse the order of the foci to get closest to ref foci
if LA.norm(foci1 - foci1_ref) > LA.norm(foci1 - foci2_ref):
    _temp = foci1
    foci1 = foci2
    foci2 = _temp

####################################################
# plot the fitted ellipse foci
pp.plot([foci1[0]], [foci1[1]], 'xk')
pp.plot([foci2[0]], [foci2[1]], 'xk')

# plot a line between the fitted ellipse foci and the reference foci
pp.plot([foci1[0], foci1_ref[0]], [foci1[1], foci1_ref[1]], 'm-')
pp.plot([foci2[0], foci2_ref[0]], [foci2[1], foci2_ref[1]], 'm-')

# plot fitted ellipse
(ellipse_x, ellipse_y) = gen_ellipse_pts(a, foci1, foci2, num_pts=1000)
pp.plot(ellipse_x, ellipse_y, 'r-', lw=3, alpha=0.5)

# scale the axes for a square display
x_max = max(x_list)
x_min = min(x_list)
y_max = max(y_list)
y_min = min(y_list)

box_max = max([x_max, y_max])
box_min = min([x_min, y_min])
pp.axis([box_min, box_max, box_min, box_max])

pp.show()