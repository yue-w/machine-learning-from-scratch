# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:57:26 2020

@author: wyue
Original tutorial: https://www.youtube.com/watch?v=s_mgVRfLysY&t=1875s
"""
import numpy as np
from matplotlib import pyplot as plt 

def data_generator():
    g1 = np.array([3,4]) + np.random.randn(100,2)
    g2 = np.array([10,-4]) + np.random.randn(100,2)
    g3 = np.array([-5,0]) + np.random.randn(100,2)
    
    return np.concatenate([g1,g2,g3], axis = 0)

def kmeans(data, K):
    N =  data.shape[0]
    D = data.shape[1]
    category = np.zeros(N,dtype='int')
    centroid = np.random.randn(K,D)
    ## Loop until converging
    FINISHED = False
    while not FINISHED:
        #print(FINISHED)
        ## for each point, set category

        for p in range(N):
            cat = None
            min_dis = float('inf')
            ## for each centroid
            for index, cen in enumerate(centroid):
                dis = np.linalg.norm(data[p]-cen)
                if dis<min_dis:
                    min_dis = dis
                    cat = index
            category[p] = cat
        centroid_change = 0
        ## Update centroid
        for k in range(K):
             centroid_new = np.mean(data[category==k], axis=0)
             centroid_change += np.linalg.norm(centroid_new-centroid[k])
             centroid[k] = centroid_new
        if centroid_change< 1e-3:
            FINISHED = True
    return centroid, category
if __name__ == '__main__':
    data = data_generator()
    #plt.scatter(data[:,0],data[:,1])
    #plt.show()
    
    centroid , category = kmeans(data, 3)
    plt.scatter(data[:,0],data[:,1], c=category)
    plt.scatter(centroid[:,0],centroid[:,1],c='r',marker='+')
    plt.show()