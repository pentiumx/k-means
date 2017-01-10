import numpy as np
import os
import sys
import random
import math

radius = 10
num = 100
#input_data = [(1, 2), (3, 4), (1, -5), (-2, 3)]
input_data = np.random.uniform(-radius, radius, (num, 2))

def kmeans(input_data, k):
    # perform k-means clustering
    # ref: https://home.deib.polimi.it/matteucc/Clustering/tutorial_html/kmeans.html

    # initialize mean vectors
    prev_means = np.random.uniform(0, 10, (k, 2))
    cur_means = prev_means

    # mapping array showing that which cluster a point belongs to
    clusters = np.random.randint(0, k, input_data.shape[0])

    # repeat until there are no changes in mean vectors
    while (prev_means != cur_means).all():
        prev_means = cur_means

        # Step 1: use the estimated means to classify data samples
        for i, data in enumerate(input_data):
            min_mean = 0 # index of mean
            min_dist = 9999  # current minimum distance

            # data => [x, y]
            # check which centroid is the nearest
            for j, mean in enumerate(cur_means):
                dist = dist(data, mean)
                if dist < min_dist:
                    min_mean = j
                    min_dist = dist

            # update the cluster this point belongs to
            clusters[i] = min_mean

        # Step 2: replace means with mean of current data samples belonging to cluster i
        for j, mean in enumerate(cur_means):
            samples = clusters[x == j]
            cur_means[j] = np.mean(samples)

    # return cluster info and means of clusters
    return clusters, cur_means

def dist(p1, p2):
    d1 = p1[0]-p2[0]
    d2 = p1[1]-p2[1]
    return math.sqrt( d1*d1 + d2*d2 )

print (input_data)
print (kmeans(input_data, 2))