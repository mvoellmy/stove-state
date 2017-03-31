
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# TODO:
# - Shannon Entropy
# - Single Trees

pad = 1

frame_test = cv2.imread('../../data/In-airGestures/Training/gesture1/NoisySegmentation/tip_noisy50.png')
img_test = frame_test[:,:,0]
H = len(img_test)
W = len(img_test[0])
np.random.seed(1)
w = np.random.randint(-pad,pad+1,H*W*2)
np.random.seed(2)
v = np.random.randint(-pad,pad+1,H*W*2)

mypath = '../../data/In-airGestures/Training/gesture1/CleanSegmentation'
mypath_noisy = '../../data/In-airGestures/Training/gesture1/NoisySegmentation'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
onlyfiles_noisy = [ f for f in listdir(mypath_noisy) if isfile(join(mypath_noisy,f)) ]
images = np.empty(len(onlyfiles), dtype=object)


print("Extracting Features...")
start = time()
X_train = []
L_train = []
for n in range(0, 100):
    frame = cv2.imread(join(mypath, onlyfiles[n]))
    frame_noisy = cv2.imread(join(mypath_noisy, onlyfiles_noisy[n]))
    img = frame[:,:,0]
    img_noisy = frame_noisy[:,:,0]
    img_noise = cv2.subtract(img, img_noisy)
    cv2.imshow("A", img)
    cv2.imshow("B", img_noisy)
    cv2.imshow("asd", img_noise)
    cv2.waitKey(0)
    for i in range(pad,H-pad):
        for j in range(pad,W-pad):
            F = [img[i+w[i+j], j+w[i+j+W*H]], img[i+v[i+j], j+v[i+j+W*H]]]
            #F = [img[i-1,j-1], img[i-1,j], img[i-1,j+1], img[i,j-1], img[i,j+1], img[i+1,j-1], img[i+1,j], img[i+1,j-1]]
            X_train.append(F)
            L_train.append(img[i,j])

X_test = []
for i in range(pad, H - pad):
    for j in range(pad, W - pad):
        F = [img_test[i+w[i+j], j+w[i+j+W*H]], img_test[i+v[i+j], j+v[i+j+W*H]]]
        #img = img_test
        #F = [img[i-1, j-1], img[i-1, j], img[i-1, j+1], img[i, j-1], img[i, j+1], img[i+1, j-1], img[i+1, j], img[i+1, j-1]]
        X_test.append(F)
print("Took %.2f seconds" % (time() - start))


print("Training...")
start = time()
clf = RandomForestClassifier(n_estimators=10, criterion="entropy", n_jobs=-1)
clf = clf.fit(X_train, L_train)
print("Took %.2f seconds" % (time() - start))
a = clf.decision_path(X_test)
print(clf.decision_path(X_test))

print("Predicting...")
start = time()
prediction = clf.predict(X_test)
print("Took %.2f seconds" % (time() - start))

img_out = np.reshape(prediction, (H-2*pad,W-2*pad))

#cv2.imshow("Image", img)
cv2.imshow("Image Noisy", img_test)
cv2.imshow("Image Output", img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
