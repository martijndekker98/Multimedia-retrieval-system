import distanceFunctions
import staticVariables as sV
import numpy as np
from sortedcontainers import SortedList


# find the most similar models given the database, index of model and a distance matrix.
# So, assumes distances are already computed (useful for evaluation)
def findMostSimilarN(database: np.ndarray, index: int, distanceMat: np.ndarray):
    ans = SortedList(key=lambda tup: tup[1])
    for i in range(0, index):
        ans.add((database[i,1], distanceMat[index, i], database[i,0]))
    for i in range(index+1, sV.modelsTotal):
        ans.add((database[i,1], distanceMat[index, i], database[i,0]))
    return ans


# Given the database, dictionary with histogram distances, dictionary with global distances, some weights:
# Return a matrix containing the distances between the models in the database
def getDistanceMatrix(database: np.ndarray, histDists: dict, globalDists: dict, gsw: list, shapeW: list):
    ans = np.zeros((len(database), len(database)))
    for i in range(0, len(database)):
        for j in range(i+1, len(database)):
            key = (database[i, 1], database[j,1])
            shapeDist = sum([shapeW[i] * histDists[key][i] for i in range(0, 5)])
            ans[i, j] = ans[j, i] = (globalDists[key] * gsw[0]) + (shapeDist * gsw[1])
    return ans


# Given a database, compute the matrix containing the distances between the models
def compDistanceMatrix(database: np.ndarray):
    ans = np.zeros((len(database), len(database)))
    for i in range(0, len(database)):
        for j in range(i+1, len(database)):
            # key = (database[i, 1], database[j,1])
            dist = distanceFunctions.compDistance(database[i], database[j])
            ans[i, j] = ans[j, i] = dist
    return ans


# Given a database and a model index: compute the distance to all other models and sort them
# Return list with models from most to least similar to model 'index'
def getQueryResults(database: np.ndarray, index: int):
    ans = SortedList(key=lambda tup: tup[1])
    for i in range(0, index):
        ans.add((database[i, 1], distanceFunctions.compDistance(database[index], database[i]), database[i, 0]))
    for i in range(index + 1, sV.modelsTotal):
        ans.add((database[i, 1], distanceFunctions.compDistance(database[index], database[i]), database[i, 0]))
    return ans