# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:50:08 2017

@author: ian
"""

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from random import randint, seed
import matplotlib.pyplot as plt

mypath='../../data/In-airGestures/Training/gesture1/CleanSegmentation'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
img = cv2.imread( join(mypath,onlyfiles[0]) )
S = (img[:,:,0] == np.zeros(img.shape[0:2])) * 1
S2 = S*2 - 1
rows, cols = img.shape[0:2]

seed(3)
Gamma = [[0,0], [1,0], [0,1], [1,1]]
w = [randint(-1,1), randint(-1,1)]
v = [randint(-1,1), randint(-1,1)]
pad = 10
num_rand = 100

vec_rand = np.zeros((num_rand,4))
F = np.zeros((rows,cols,2), dtype='uint8')
C = np.zeros((rows,cols,4), dtype='uint8')
C_count = np.zeros((num_rand,4))

for k in range(0,num_rand):
    w = [randint(-pad,pad), randint(-pad,pad)]
    v = [randint(-pad,pad), randint(-pad,pad)]
    vec_rand[k,0:2] = w
    vec_rand[k,2:4] = v
    F = np.zeros((rows,cols,2), dtype='uint8')
    C = np.zeros((rows,cols,4), dtype='uint8')
    for i in range(pad,rows-pad):
        for j in range(pad,cols-pad):
            F[i, j, 0] = S[i + w[0], j + w[1]]
            F[i, j, 1] = S[i + v[0], j + v[1]]
            if [F[i, j, 0], F[i, j, 1]] == Gamma[0]:
                C[i,j,0] = 1
                C_count[k,0] = C_count[k,0] + 1*S2[i,j]
            elif [F[i, j, 0], F[i, j, 1]] == Gamma[1]:
                C[i,j,1] = 1
                C_count[k,1] = C_count[k,1] + 1*S2[i,j]
            elif [F[i, j, 0], F[i, j, 1]] == Gamma[2]:
                C[i,j,2] = 1
                C_count[k,2] = C_count[k,2] + 1*S2[i,j]
            elif [F[i, j, 0], F[i, j, 1]] == Gamma[3]:
                C[i,j,3] = 1
                C_count[k,3] = C_count[k,3] + 1*S2[i,j]
    C2 = (C*np.float(1))*2-1
    C_count[k,0] = (C2[pad:-pad,pad:-pad,0]*S2[pad:-pad,pad:-pad]).sum()
    C_count[k,1] = (C2[pad:-pad,pad:-pad,1]*S2[pad:-pad,pad:-pad]).sum()
    C_count[k,2] = (C2[pad:-pad,pad:-pad,2]*S2[pad:-pad,pad:-pad]).sum()
    C_count[k,3] = (C2[pad:-pad,pad:-pad,3]*S2[pad:-pad,pad:-pad]).sum()

#%% Take features with most information gain
n_nodes = 7
C = np.zeros((7,6))
C_count_abs = np.abs(C_count)
for i in range(0,n_nodes):
    max_idx = C_count_abs.argmax()
    max_row = int(max_idx / C_count_abs.shape[1])
    max_col = max_idx % C_count_abs.shape[1]
    print("row: %s, col: %s" % (max_row, max_col))
    C_count_abs[max_row, max_col] = 0
    C[i,0:4] = vec_rand[max_row,:]
    C[i,4:6] = Gamma[max_col]

#%% Train tree

leaf_counts = np.zeros((2,8))


for i in range(pad,rows-pad):
    for j in range(pad,cols-pad):
        w = C[0, 0:2]
        v = C[0, 2:4]
        G = C[0, 4:6]
        G = G.tolist()
        F = [S[i + w[0], j + w[1]], S[i + v[0], j + v[1]]]
        if G == F:
            w = C[1, 0:2]
            v = C[1, 2:4]
            G = C[1, 4:6]
            G = G.tolist()
            F = [S[i + w[0], j + w[1]], S[i + v[0], j + v[1]]]
            if G == F:
                w = C[2, 0:2]
                v = C[2, 2:4]
                G = C[2, 4:6]
                G = G.tolist()
                F = [S[i + w[0], j + w[1]], S[i + v[0], j + v[1]]]
                if G == F:
                    if S[i,j] == 0:
                        leaf_counts[0,0] += 1
                    else:
                        leaf_counts[1,0] += 1
                else:
                    if S[i,j] == 0:
                        leaf_counts[0,1] += 1
                    else:
                        leaf_counts[1,1] += 1
            else:
                w = C[3, 0:2]
                v = C[3, 2:4]
                G = C[3, 4:6]
                G = G.tolist()
                F = [S[i + w[0], j + w[1]], S[i + v[0], j + v[1]]]
                if G == F:
                    if S[i,j] == 0:
                        leaf_counts[0,2] += 1
                    else:
                        leaf_counts[1,2] += 1
                else:
                    if S[i,j] == 0:
                        leaf_counts[0,3] += 1
                    else:
                        leaf_counts[1,3] += 1
        else:
            w = C[4, 0:2]
            v = C[4, 2:4]
            G = C[4, 4:6]
            G = G.tolist()
            F = [S[i + w[0], j + w[1]], S[i + v[0], j + v[1]]]
            if G == F:
                w = C[5, 0:2]
                v = C[5, 2:4]
                G = C[5, 4:6]
                G = G.tolist()
                F = [S[i + w[0], j + w[1]], S[i + v[0], j + v[1]]]
                if G == F:
                    if S[i,j] == 0:
                        leaf_counts[0,4] += 1
                    else:
                        leaf_counts[0,4] += 1
                else:
                    if S[i,j] == 0:
                        leaf_counts[0,5] += 1
                    else:
                        leaf_counts[1,5] += 1
            else:
                w = C[6, 0:2]
                v = C[6, 2:4]
                G = C[6, 4:6]
                G = G.tolist()
                F = [S[i + w[0], j + w[1]], S[i + v[0], j + v[1]]]
                if G == F:
                    if S[i,j] == 0:
                        leaf_counts[0,6] += 1
                    else:
                        leaf_counts[1,6] += 1
                else:
                    if S[i,j] == 0:
                        leaf_counts[0,7] += 1
                    else:
                        leaf_counts[1,7] += 1
        
#%% Test sample image
img_input = cv2.imread('../../data/In-airGestures/Training/gesture1/NoisySegmentation/tip_noisy1.png')
img_output = np.zeros((rows,cols,2), dtype='uint8')
A = (img_input[:,:,0] == np.zeros(img_input.shape[0:2])) * 1
B = img_output
for i in range(pad,rows-pad):
    for j in range(pad,cols-pad):
        w = C[0, 0:2]
        v = C[0, 2:4]
        G = C[0, 4:6]
        G = G.tolist()
        F = [A[i + w[0], j + w[1]], A[i + v[0], j + v[1]]]
        if G == F:
            w = C[1, 0:2]
            v = C[1, 2:4]
            G = C[1, 4:6]
            G = G.tolist()
            F = [A[i + w[0], j + w[1]], A[i + v[0], j + v[1]]]
            if G == F:
                w = C[2, 0:2]
                v = C[2, 2:4]
                G = C[2, 4:6]
                G = G.tolist()
                F = [A[i + w[0], j + w[1]], A[i + v[0], j + v[1]]]
                if G == F:
                    B[i,j] = 0
                else:
                    B[i,j] = 0
            else:
                w = C[3, 0:2]
                v = C[3, 2:4]
                G = C[3, 4:6]
                G = G.tolist()
                F = [A[i + w[0], j + w[1]], A[i + v[0], j + v[1]]]
                if G == F:
                    B[i,j] = 0
                else:
                    B[i,j] = 0
        else:
            w = C[4, 0:2]
            v = C[4, 2:4]
            G = C[4, 4:6]
            G = G.tolist()
            F = [A[i + w[0], j + w[1]], A[i + v[0], j + v[1]]]
            if G == F:
                w = C[5, 0:2]
                v = C[5, 2:4]
                G = C[5, 4:6]
                G = G.tolist()
                F = [A[i + w[0], j + w[1]], A[i + v[0], j + v[1]]]
                if G == F:
                    B[i,j] = 0
                else:
                    B[i,j] = 255
            else:
                w = C[6, 0:2]
                v = C[6, 2:4]
                G = C[6, 4:6]
                G = G.tolist()
                F = [A[i + w[0], j + w[1]], A[i + v[0], j + v[1]]]
                if G == F:
                    B[i,j] = 255
                else:
                    B[i,j] = 0


#%% Display images
#cv2.imshow("Image", img)
plt.figure()
plt.imshow(S)
plt.figure()
plt.imshow(A)
plt.figure()
plt.imshow(B[:,:,0])
"""
cv2.imshow("Feature Response", F[:,:,0]*255)
cv2.imshow("C0", C[:,:,0]*255)
cv2.imshow("C1", C[:,:,1]*255)
cv2.imshow("C2", C[:,:,2]*255)
cv2.imshow("C3", C[:,:,3]*255)

cv2.imshow("input", A)
cv2.imshow("output", B)
cv2.waitKey(0) 
cv2.destroyAllWindows()
"""