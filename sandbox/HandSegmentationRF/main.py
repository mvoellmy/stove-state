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
from time import time
seed(1)

# Using random parameters get those with most information gain
def getBestParam(num_samples, S, pad):
    rows, cols = S.shape
    gammas = [[0,0], [1,0], [0,1], [1,1]]
    
    gamma_rand = [gammas[randint(0,3)] for i in range(0,num_samples)]
    w_rand = [[randint(-pad,pad), randint(-pad,pad)] for i in range(0,num_samples)]
    v_rand = [[randint(-pad,pad), randint(-pad,pad)] for i in range(0,num_samples)]
    F = np.zeros((rows,cols,2,num_samples), dtype='uint8')
    C = np.zeros((rows,cols,num_samples), dtype='int')
    C_count = np.zeros((2,num_samples))
    C_count_abs = np.zeros((2,num_samples))
    for k in range(0, num_samples):
        
        for i in range(pad, rows-pad):
            for j in range(pad, cols-pad):
                F[i,j,0,k] = S[i+w_rand[k][0], j+w_rand[k][1]]
                F[i,j,1,k] = S[i+v_rand[k][0], j+v_rand[k][1]]
                C[i,j,k] = ([F[i,j,0,k], F[i,j,1,k]] == gamma_rand[k])*1
#        cv2.imshow("asd",C[pad:-pad,pad:-pad,k].astype(np.uint8)*255)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        C_count[0,k] = abs((C[pad:-pad,pad:-pad,k]*(S[pad:-pad,pad:-pad]-1)).sum())
        C_count[1,k] = (C[pad:-pad,pad:-pad,k]*S[pad:-pad,pad:-pad]).sum()
        if C_count[0,k] > 0 and C_count[1,k] > 0:
            C_count_abs[0,k] = C_count[0,k]/(C_count[0,k] + C_count[1,k])
            C_count_abs[1,k] = C_count[1,k]/(C_count[0,k] + C_count[1,k])
#        else:
#            C_count_abs[0,k] = 0.5
#            C_count_abs[1,k] = 0.5
#    norm = C_count.sum(0)
#    C_count_abs[0,:] = C_count[0,:]/norm
#    C_count_abs[1,:] = C_count[1,:]/norm
    H = np.zeros(C_count.shape[1])
    for i in range(0,C_count_abs.shape[1]):
        if C_count_abs[0,i] > 0 and C_count_abs[1,i] > 0:
            H[i] = - C_count_abs[0,i]*np.log(C_count_abs[0,i]) - C_count_abs[1,i]*np.log(C_count_abs[1,i])
    idx_best = np.argmax(-H)
    
    return w_rand[idx_best], v_rand[idx_best], gamma_rand[idx_best]#, C_count_abs, H, idx_best, C_count

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
        
def traverseTree(Tree, tree_depth, S, i, j):
    #text_file.write("Pixel [%s, %s]: " % (i, j))
    current_node = Tree[0]
    for k in range(0,tree_depth):
        w = current_node.data[0].astype(np.int)
        v = current_node.data[1].astype(np.int)
        gamma = current_node.data[2]
        F = [S[i+w[0], j+w[1]], S[i+v[0], j+v[1]]]
        G = [int(gamma[0]), int(gamma[1])]
        #text_file.write("F: %s, G: %s " % (F, G))
        if current_node.children != []:
            if F == G:
                current_node = current_node.children[0]
                #text_file.write("L, ")
            else:
                current_node = current_node.children[1]
                #text_file.write("R, ")
        else:
            idx_branch = current_node.data[3] - (2**(tree_depth-1) - 1)
            if F == G:
                idx_leaf = 0
                #text_file.write("L\n")
            else:
                idx_leaf = 1
                #text_file.write("R\n")

    return idx_leaf, idx_branch

#%% Test
start = time()
tree_depth = 2
num_nodes = 2**(tree_depth)-1
num_leaves = num_nodes + 1
num_trees = 1
num_samples = 20
pad = 10

img = cv2.imread('../../data/In-airGestures/Training/gesture1/CleanSegmentation/tip1.png')
S = (img[:,:,0] == np.zeros(img.shape[0:2])) * 1

#w, v, gamma, C, H, idx, C2 = getBestParam(num_samples, S, pad)
#print(gamma[idx])
 
#%% Build Random Forest
Forest = []

for t in range(0,num_trees):
    # Get best parameters for split nodes
    w_vec = np.zeros((2,num_nodes))
    v_vec = np.zeros((2,num_nodes))
    gamma_vec = np.zeros((2,num_nodes))
    for i in range(0,num_nodes):
        w, v, gamma = getBestParam(num_samples, S, pad)
        w_vec[:,i] = np.array(w)
        v_vec[:,i] = np.array(v)
        gamma_vec[:,i] = np.array(gamma)
    
    # Build Tree 
    # Node data: w, v, gamma, idx, leaf_distribution
    Tree = [Node((w_vec[:,i],v_vec[:,i],gamma_vec[:,i],i,np.zeros((2,2)))) for i in range(0,num_nodes)]   
    node_counter = 1  
    for i in range(0,2**(tree_depth-1)-1):
        for j in range(0,2):
            Tree[i].add_child(Tree[node_counter+j])
        node_counter += 2
    
    # Fill Tree
    leaf_values = np.zeros((num_leaves,2))
    #text_file = open("output.txt", "w")
    for i in range(pad,S.shape[0]-pad):
        for j in range(pad,S.shape[1]-pad):
            idx_leaf, idx_branch = traverseTree(Tree, tree_depth, S, i, j)
            if S[i,j] == 0:
                Tree[idx_branch].data[4][idx_leaf][0] += 1
            else:
                Tree[idx_branch].data[4][idx_leaf][1] += 1
                #Tree[idx_branch].data[4][idx_leaf2] += 1
    
    #text_file.close()
    Forest.append(Tree)
    
#%%
text_file = open("forest.txt", "w")

for idx_tree, tree in enumerate(Forest):
    print("Tree #%s\n" % idx_tree)
    text_file.write("Tree #%s\n" % idx_tree)
    node_i = 0
    for idx, node in enumerate(tree):
        print("Gamma: %s, Distrib: %s" % (node.data[2], node.data[4]))
    for i in range(0,tree_depth):
        for j in range(0,2**i):
            text_file.write("%s \t" % (tree[node_i].data[2]))
            node_i += 1
        text_file.write("\n")
    text_file.write("\n")
            
text_file.close()    
#%% Evaluate noisy image
#plt.ion()
plt.close("all")
img_n = cv2.imread('../../data/In-airGestures/Training/gesture1/NoisySegmentation/tip_noisy1.png')
S_n = (img_n[:,:,0] == np.zeros(img_n.shape[0:2])) * 1

S_final = np.zeros((S_n.shape[0],S_n.shape[1]))
#text_file = open("output.txt", "w")
for i in range(pad,S_n.shape[0]-pad):
    for j in range(pad,S_n.shape[1]-pad):   
        probabilities = np.zeros((num_trees,2))
        for idx, tree in enumerate(Forest):
            idx_leaf, idx_branch = traverseTree(tree, tree_depth, S_n, i, j)
            probabilities[idx] = tree[idx_branch].data[4][idx_leaf]        
        S_final[i,j] = np.argmax([probabilities[:,0].sum(), probabilities[:,1].sum()])
#text_file.close()

fig = plt.figure(2)
plt.subplot(121), plt.imshow(S_n)        
plt.subplot(122), plt.imshow(S_final)
fig.savefig('%sTrees-Depth%s-pad%s' % (num_trees, tree_depth, pad))

print("Took %s min %s sec" % (int((time()-start)/60), int((time()-start)%60)))

        
