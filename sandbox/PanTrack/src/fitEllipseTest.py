import numpy as np
from numpy.linalg import eig, inv
from fitEllipse import *

arc = .5

R = np.arange(0,arc*np.pi, 0.01)
x = 1.5*np.cos(R) + 10 + 0.1*np.random.rand(len(R))
y = np.sin(R) + 5 + 0.1*np.random.rand(len(R))

a = fitEllipse(x,y)
center = ellipse_center(a)
#phi = ellipse_angle_of_rotation(a)
phi = ellipse_angle_of_rotation2(a)
axes = ellipse_axis_length(a)

print("center = ",  center)
print("angle of rotation = ",  phi)
print("axes = ", axes)


arc_ = 2
R_ = np.arange(0,arc_*np.pi, 0.01)
a, b = axes
xx = center[0] + a*np.cos(R_)*np.cos(phi) - b*np.sin(R_)*np.sin(phi)
yy = center[1] + a*np.cos(R_)*np.sin(phi) + b*np.sin(R_)*np.cos(phi)

from pylab import *
plot(x,y)
plot(xx,yy, color = 'red')
show()
