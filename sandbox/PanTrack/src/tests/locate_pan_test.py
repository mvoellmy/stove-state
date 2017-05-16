import cv2
from panlocator import PanLocator

# Read Images
img_path = '../../../../data/stills/stove_left_on.PNG'
img_path = '../../../../data/stills/stove_left_on.PNG'
img_path = '../../../../data/stills/stove_top.PNG'
img_path = '../../../../data/stills/hot_butter_in_pan.PNG'
img_path = '../../../../data/stills/boiling_water.PNG'
img_path = '../../../../data/stills/frame-000055.jpg'

img = cv2.imread(img_path)

pan_locator = PanLocator()

cv2.imshow('img', img)
pan_locator.find_pan(img)
