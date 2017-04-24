import cv2
from panfind import panfind

# Read Images
img_path = '../../data/stills/stove_left_on.PNG'
img_path = '../../data/stills/stove_left_on.PNG'
img_path = '../../data/stills/stove_top.PNG'
img_path = '../../data/stills/frame-000055.jpg'
img_path = '../../data/stills/boiling_water.PNG'
img_path = '../../data/stills/hot_butter_in_pan.PNG'

img = cv2.imread(img_path)

panfind(img,rgb=True, _plot_ellipse=True)