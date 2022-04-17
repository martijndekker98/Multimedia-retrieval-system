import os
from sklearn.neighbors import NearestNeighbors, KDTree
import step4
import pandas.core.frame
import numpy as np
import time
import pickle
import sys
from collections import defaultdict


def ann(query, k, r):

    # kneigh = neighbors.kneighbors(X=query, n_neighbors=k)
    rneigh = neighbors.radius_neighbors(X=query, radius=r)

    rneigh_values = rneigh[0][0].tolist()
    rneigh_indices = rneigh[1][0].tolist()

    sorted_indices = [rneigh_indices for _,rneigh_indices in sorted(zip(rneigh_values,rneigh_indices))]
    sorted_values = sorted(rneigh_values)

    nr_of_rneigh = len(rneigh[0][0])

    k_result = {}
    r_result = {}
    # for i in range(k):
    #     k_result[i] = [kneigh[1][0][i], kneigh[0][0][i]]

    for i in range(nr_of_rneigh):
        r_result[i] = [sorted_indices[i], sorted_values[i]]


    return k_result, r_result


def make_kdtree():

    dataframe = step4.readCSVAsDataFrame('./featuresNew.csv')
    dataframe_np = dataframe.to_numpy()
    dataframe_np_head = dataframe_np
    dataframe_np = dataframe_np[:,2:]


    tree =  KDTree(dataframe_np)

    with open('knn_tree.pickle', 'wb') as handle:
        pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)


def query_kdtree(dataframe, query, k_or_r, v):

    dataframe_np = dataframe.to_numpy()
    dataframe_np_head = dataframe_np
    dataframe_np = dataframe_np[:,2:]

    query = dataframe.loc[dataframe['File_name'] == query]
    query_np = query.to_numpy()[0][2:].reshape(1,-1)

    with open('knn_tree.pickle', 'rb') as handle:
        tree = pickle.load(handle)

    if k_or_r == 1:
        dist, ind = tree.query(query_np, k=int(v)+1)
        points = dataframe_np_head[ind[0]]
        print(ind)
        print(points)
        dist = dist[0]
    elif k_or_r == 2:
        ind, dist = tree.query_radius(query_np, r=float(v), return_distance=True, sort_results=True)
        if len(ind[0]) > 1:
            points = dataframe_np_head[ind[0]]
            dist = dist[0]

    similar_shapes = []

    nr_of_neighbors = len(dist)
    if nr_of_neighbors < 6:
        for i in range(1,nr_of_neighbors):
            similar_shapes.append((points[i][1], dist[i]))
        # for i in range(nr_of_neighbors, 6):
        #     similar_shapes.append(('None', 'None'))
    else:
        for i in range(1,6):
            similar_shapes.append((points[i][1], dist[i]))

    # print(similar_shapes)
    return similar_shapes



# make_kdtree()

