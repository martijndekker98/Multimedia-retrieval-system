import pstats
import queue
import time
from copy import deepcopy

import pandas
import roc
import staticVariables as sV
import step4
import numpy as np
import pickle
import distanceFunctions
import cProfile
import heapq
import queryHelpFunctions as qHF


# Category list is used throughout file
df = step4.readCSVAsDataFrame('featuresNew.csv')
category_list = roc.getCategoryList(df)
# database, dataframe col 0 -> index in category list

# Make a dictionary with distances between models and save them in a pickle file
def pickleDistancesWithoutWeights(dataframe: pandas.DataFrame, fileName: str):
    dict_ = dict()
    for id1 in range(0, sV.modelsTotal):
        print(f"Id1: {id1}")
        for id2 in range(id1+1, sV.modelsTotal):
            feats1 = dataframe.iloc[id1, :]
            feats2 = dataframe.iloc[id2, :]
            shapeDistances = distanceFunctions.compDistanceShape(feats1, feats2)
            dict_[(feats1.iloc[1], feats2.iloc[1])] = shapeDistances
            dict_[(feats2.iloc[1], feats1.iloc[1])] = shapeDistances

    outfile = open(fileName, 'wb')
    pickle.dump(dict_, outfile)
    outfile.close()
    print("Creating pickle file is done")


# Compute the confusion matrix given the list of mostSimilar models, query size and the index of the category of the queried model
def computeConfMat(mostSimilar, querySize: int, catIndex: int):
    conf_mat = np.zeros((2, 2), dtype=np.int32)
    for i in range(0, querySize):
        conf_mat[0 if mostSimilar[i][2] == catIndex else 1, 0] += 1
    conf_mat[0, 1] += 19 - conf_mat[0,0]
    conf_mat[1, 1] += sV.maxModelsQuery - 19 - (querySize - conf_mat[0,0])
    if conf_mat[0,0]+conf_mat[0,1]+conf_mat[1,0]+conf_mat[1,1] != 379:
        print(f"Total != 379, {conf_mat}")
    return conf_mat


# Get a dictionary containg the distances of gloval features of the models
def getGlobalDistDict(database: np.ndarray, globalW):
    dict_ = dict()
    for i in range(0, len(database)):
        model1 = database[i, sV.startGlobal:sV.endGlobal]
        for j in range(i+1, len(database)):
            dist = distanceFunctions.compDistanceGlobalpart(model1, database[j, sV.startGlobal:sV.endGlobal], sV.useEuclidean, sV.useAngular, globalW, sV.useWeightedVersion)
            # dist = 0.0
            dict_[(database[i,1], database[j,1])] = dist
            dict_[(database[j,1], database[i,1])] = dist
    return dict_


# Try out some weights and compute the performance
def tryWeights(histDists: dict, querySizes: list, globalDists: dict, gsw: list, shapeW: list, database: np.ndarray):
    # Met die weights een matrix aanmaken o.b.v. histDists en die dan gebruiken voor findmostSimilar etc.
    distanceMat = qHF.getDistanceMatrix(database, histDists, globalDists, gsw, shapeW)
    precisions = [0]*len(querySizes)
    recalls = [0]*len(querySizes)
    f1s = [0]*len(querySizes)
    confMats = []
    for q in querySizes:
        confMats.append(np.zeros((2,2), dtype=np.int32))
    for i in range(0, len(database)):
        catIndex = database[i,0]
        mostSimilar = qHF.findMostSimilarN(database, i, distanceMat)
        for qID, qSize in enumerate(querySizes):
            confMat = computeConfMat(mostSimilar, qSize, catIndex)
            confMats[qID] += confMat
    for qID, qSize in enumerate(querySizes):
        precision = confMats[qID][0,0] / (confMats[qID][0,0] + confMats[qID][1,0])
        recall = confMats[qID][0,0] / (confMats[qID][0,0] + confMats[qID][0,1])
        precisions[qID] += precision
        recalls[qID] += recall
        f1s[qID] += (2 * precision * recall) / (precision + recall)
    # for q in range(0, len(querySizes)):
    #     print(f"{confMats[q]}, pre: {precisions[q]}, rec: {recalls[q]}, f1: {f1s[q]}")
    return precisions, recalls, f1s



# GSWeights: will try all the weights for global, 1-weights for shape
# Find the best parameter for the database
def findBestParameters(database: np.ndarray, histDists: dict, querySizes: list, GSWeights: list, globalWeights: list, shapeWeights: list):
    precisions = [(0, [])]*len(querySizes)
    recalls = [(0, [])]*len(querySizes)
    f1s_ = [(0, [])]*len(querySizes)
    for gw in globalWeights:
        for a in range(0, 5):
            globalW = [1.0]*5
            globalW[a] = gw
            globalDistDict = getGlobalDistDict(database, globalW)
            for sw in shapeWeights:
                for b in range(0, 5):
                    shapeW = [1.0]*5
                    shapeW[b] = sw
                    for gsw in GSWeights:
                        GSW = [gsw, 1-gsw]
                        precs, recs, f1s = tryWeights(histDists, querySizes, globalDistDict, GSW, shapeW, database)
                        print(f"Weights, gsw: {GSW}, global: {globalW}, shape: {shapeW}")
                        for qID, q in enumerate(querySizes):
                            print(f"Q_size: {q}, precision: {precs[qID]}, recall: {recs[qID]}, f1: {f1s[qID]}")
                            if precs[qID] > precisions[qID][0]:
                                precisions[qID] = (precs[qID], [globalW, shapeW, gsw])
                            if recs[qID] > recalls[qID][0]:
                                recalls[qID] = (recs[qID], [globalW, shapeW, gsw])
                            if f1s[qID] > f1s_[qID][0]:
                                f1s_[qID] = (f1s[qID], [globalW, shapeW, gsw])
                        print("~~~~~~~~~~~~")
    print("The best:")
    for q in range(0, len(querySizes)):
        print(f"Q: {querySizes[q]}, The best precision: {precisions[q]}")
        print(f"Q: {querySizes[q]}, The best recall: {recalls[q]}")
        print(f"Q: {querySizes[q]}, The best f1: {f1s_[q]}")


# Return the neighbours (in terms of weights)
def getNeighbours(current: list, GSWUp: list, globalUp: list, shapeUp: list, visited: dict, roundingOff: int = 2):
    ans = []
    # GSW
    for mode in range(0, 2):
        for i in range(0, len(GSWUp[mode])):
            x = current[0][0] + GSWUp[mode][i] if mode == 0 else current[0][0] * GSWUp[mode][i]
            if 1 >= x >= 0:
                x = round(x, roundingOff)
                a = ((x, 1 - x), deepcopy(current[1]), deepcopy(current[2]))
                try:
                    _ = visited[a]
                except KeyError:
                    # print(f"x = {x}, a = {a}")
                    ans.append(a)
    # global or shape
    for gOrS in range(0, 2):                            # Update global or shape
        array = globalUp if gOrS == 0 else shapeUp      # Pick the right list
        # print(f"Global: {gOrS == 0}")
        for mode in range(0, 2):                        # For the mode/sublist in array (add or multiply
            for i in range(0, len(array[mode])):        # For the possible value to add/multiply with
                for index in range(0, len(current[gOrS+1])):    # For the parameter
                    curSubEl = current[gOrS+1][index]           # The parameter curently
                    x = curSubEl + array[mode][i] if mode == 0 else curSubEl * array[mode][i]
                    if x >= 0.01: # if x >= 0:
                        nieuweSub = updateTuple(current[gOrS+1], x, index, roundingOff)
                        a = (deepcopy(current[0]), nieuweSub, deepcopy(current[2])) if gOrS == 0 else \
                            (deepcopy(current[0]), deepcopy(current[1]), nieuweSub)
                        try:
                            _ = visited[a]
                        except KeyError:
                            # print(f"x = {x}, a = {a}")
                            ans.append(a)
    return ans


# Update a tuple (a list of weights) so that the sum == 1
def updateTuple(old: tuple, element: float, index: int, roundingOff: int = 2):
    som = sum(old) - old[index] + element
    # print(f"Old: {old}, index: {index}, el: {element}, som: {som}")
    # if index == 0: return (element/som, old[1]/som, old[2]/som, old[3]/som, old[4]/som)
    # elif index == 1: return (old[0]/som, element/som, old[2]/som, old[3]/som, old[4]/som)
    # elif index == 2: return (old[0]/som, old[1]/som, element/som, old[3]/som, old[4]/som)
    # elif index == 3: return (old[0]/som, old[1]/som, old[2]/som, element/som, old[4]/som)
    # elif index == 4: return (old[0]/som, old[1]/som, old[2]/som, old[3]/som, element/som)
    # else: print(f"ERROR, length of tuple: {old}")
    lijst = [x for x in old]
    lijst[index] = element
    return tuple(round(x/som, roundingOff) for x in lijst)


# Get the dictionary containing the global distances
def findGlobalDistDicts(globalDistDicts: dict, globalW: tuple, database: np.ndarray):
    som = sum(globalW)
    globalWNew = tuple([x/som for x in globalW])
    try:
        return globalDistDicts[globalWNew]
    except KeyError:
        gdd = getGlobalDistDict(database, globalWNew)
        globalDistDicts[globalWNew] = gdd
        return gdd


# Find good parameters using some sort of hillclimbing
def findParametersHillClimbing(database: np.ndarray, histDists: dict, querySizes: list, GSWUp: list, globalUp: list,
                               shapeUp: list, initial: list = [(0.5, 0.5), (1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0, 1.0)],
                               maxRandomWalk: int = 2):
    visited = dict()
    globalDistDicts = dict()
    wachtrij = queue.Queue()

    # Take care of initial point
    initial = [tuple(x) for x in initial]
    wachtrij.put((initial[0], initial[1], initial[2]))
    gdd = findGlobalDistDicts(globalDistDicts, initial[1], database)
    precs, recs, f1s = tryWeights(histDists, querySizes, gdd, initial[0], initial[2], database)
    visited[(initial[0], initial[1], initial[2])] = (0, precs, recs, f1s)
    print(f"Initial {initial}\nF1s: {f1s}")

    # Walk over the neighbours
    print("Start walking")
    maxVal = sum(f1s)
    maxWeights = initial
    # i = 0
    while not wachtrij.empty():
        current = wachtrij.get()
        currentEntry = visited[current]
        somF1s = sum(currentEntry[3])

        # checked in getNeighbours if already visited
        neighbours = getNeighbours(current, GSWUp, globalUp, shapeUp, visited)
        for neighb in neighbours:
            gdd = findGlobalDistDicts(globalDistDicts, neighb[1], database)
            precs, recs, f1s = tryWeights(histDists, querySizes, gdd, neighb[0], neighb[2], database)

            print(f"Neighb: {neighb}, len queue {wachtrij.qsize()}")
            print(f"f1s: {f1s}, combined: {sum(f1s)}")
            score = 0 if sum(f1s) > somF1s else currentEntry[0] + 1
            visited[neighb] = (score, precs, recs, f1s)
            if score <= maxRandomWalk:
                wachtrij.put(neighb)
            if sum(f1s) > maxVal:
                maxVal = sum(f1s)
                maxWeights = neighb
            # i += 1
    outfile = open('hillClimbResults.pickle', 'wb')
    pickle.dump(visited, outfile)
    pickle.dump(globalDistDicts, outfile)
    outfile.close()
    print(f"Max found: {maxVal}\nWeights: {maxWeights}")
    print(visited[maxWeights])


# Find good parameters using some sort of hill climbing in a greedy manner
# Best first search approach: neighbour with best performance is explored first
def findParamGreedyHillClimb(database: np.ndarray, histDists: dict, querySizes: list, GSWUp: list, globalUp: list,
                             shapeUp: list, initial: list = [(0.5, 0.5), (0.2, 0.2, 0.2, 0.2, 0.2), (0.2, 0.2, 0.2, 0.2, 0.2)],
                             maxRandomWalk: int = 2):
    visited = dict()
    globalDistDicts = dict()
    wachtrij = []

    # Take care of initial point
    initial = [tuple(x) for x in initial]
    gdd = findGlobalDistDicts(globalDistDicts, initial[1], database)
    precs, recs, f1s = tryWeights(histDists, querySizes, gdd, initial[0], initial[2], database)
    visited[(initial[0], initial[1], initial[2])] = (0, precs, recs, f1s)
    wachtrij.append((-1*(sum(f1s)), (initial[0], initial[1], initial[2])))
    print(f"Initial {initial}\nF1s: {f1s}")

    # Walk over the neighbours
    print("Start walking")
    maxVal = sum(f1s)
    maxWeights = initial
    heapq.heapify(wachtrij)
    # i = 0
    while wachtrij: # while wachtrij and i < 100:
        # print(wachtrij)
        current = heapq.heappop(wachtrij)[1]
        currentEntry = visited[current]
        somF1s = sum(currentEntry[3])

        # checked in getNeighbours if already visited
        neighbours = getNeighbours(current, GSWUp, globalUp, shapeUp, visited)
        for neighb in neighbours:
            gdd = findGlobalDistDicts(globalDistDicts, neighb[1], database)
            precs, recs, f1s = tryWeights(histDists, querySizes, gdd, neighb[0], neighb[2], database)

            print(f"Neighb: {neighb}, len queue {len(wachtrij)}")
            print(f"f1s: {f1s}, combined: {sum(f1s)}")
            score = 0 if sum(f1s) > somF1s else currentEntry[0] + 1
            visited[neighb] = (score, precs, recs, f1s)
            if score <= maxRandomWalk:
                heapq.heappush(wachtrij, (-1*(sum(f1s)) ,neighb))
            if sum(f1s) > maxVal:
                maxVal = sum(f1s)
                maxWeights = neighb
            # i += 1
    outfile = open('hillClimbResults.pickle', 'wb')
    pickle.dump(visited, outfile)
    pickle.dump(globalDistDicts, outfile)
    outfile.close()
    print(f"Max found: {maxVal}\nWeights: {maxWeights}")
    print(visited[maxWeights])


# Turn the pandas dataframe into an NP database for speed
def getDatabase(dataframe: pandas.DataFrame):
    db = dataframe.to_numpy(copy=True)
    for rij in db:
        rij[0] = category_list.index(rij[0])
    return db


#
#
# Below for testing only
#
#



def kortTestje():
    # tryWeights(dataframe: pandas.DataFrame, histDists: dict, querySizes: list, globalDists: dict, gsw: list, shapeW: list):
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    database = getDatabase(dataframe)
    histDists = roc.unpickleDict(sV.histDistPickle)
    querySizes = [5, 10, 20, 40]
    print("Calc the global distances")
    t1 = time.perf_counter_ns()
    globalDistDict = getGlobalDistDict(database, [1.0, 1.0, 1.0, 1.0, 1.0])
    t2 = time.perf_counter_ns()
    print("Start trying weights")
    precs, recs, f1s = tryWeights(histDists, querySizes, globalDistDict, [0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 2.0], database)
    t3 = time.perf_counter_ns()
    print(f"Time getGlobalDist: {(t2-t1)/1000000}")
    print(f"Time tryweights: {(t3-t2)/1000000}")
    for qID, q in enumerate(querySizes):
        print(f"Precision: {precs[qID]}, recall: {recs[qID]}, f1s: {f1s[qID]}")


def mainCprofile():
    # cProfile.run('kortTestje()', sort='cumulative')
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    database = dataframe.to_numpy(copy=True)
    profiler = cProfile.Profile()
    profiler.enable()
    # globalDistDict = getGlobalDistDict(database, [1.0, 1.0, 1.0, 1.0, 1.0])
    kortTestje()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()


def main():
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    histDists = roc.unpickleDict(sV.histDistPickle)
    print("Start finding parameters")
    database = dataframe.to_numpy(copy=True)
    findBestParameters(database, histDists, [5, 10, 20, 40], [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], [0.5, 2.0], [0.5, 2.0])


def hillClimbMain():
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    database = getDatabase(dataframe)
    histDists = roc.unpickleDict(sV.histDistPickle)
    # findParamGreedyHillClimb(database, histDists, [10, 20], [[], []], [[-0.01], []],
    #                          [[], []], initial= [(0.65, 0.35), (0.47, 0.39, 0.06, 0.01, 0.08), (0.16, 0.21, 0.31, 0.21, 0.11)], maxRandomWalk=0)
    # findParamGreedyHillClimb(database, histDists, [10, 20], [[0.05, -0.05, 0.02, -0.02, 0.01, -0.01], []], [[0.05, -0.05, 0.02, -0.02, 0.01, -0.01], []],
    #                          [[0.05, -0.05, 0.02, -0.02, 0.01, -0.01], []], initial= [(0.65, 0.35), (0.47, 0.39, 0.06, 0.01, 0.08), (0.16, 0.21, 0.31, 0.21, 0.11)], maxRandomWalk=0)
    # findParamGreedyHillClimb(database, histDists, [10, 20], [[0.1, -0.1, 0.05, -0.05, 0.02, -0.02], []], [[0.1, -0.1, 0.05, -0.05, 0.02, -0.02], [0.5, 2.0]],
    #                          [[0.1, -0.1, 0.05, -0.05], [0.5, 2.0]], initial= [(0.7, 0.3), (0.46, 0.38, 0.06, 0.02, 0.08), (0.16, 0.29, 0.29, 0.24, 0.02)], maxRandomWalk=0)
    findParametersHillClimbing(database, histDists, [10, 20], [[0.1, -0.1, 0.05, -0.05, 0.02, -0.02], []], [[0.1, -0.1, 0.05, -0.05, 0.02, -0.02], []],
                               [[0.1, -0.1, 0.05, -0.05, 0.02, -0.02], []], initial=[(0.65, 0.35), (0.47, 0.39, 0.06, 0.01, 0.08), (0.16, 0.21, 0.31, 0.21, 0.11)], maxRandomWalk=0)
    # findParametersHillClimbing(database, histDists, [10, 20], [[0.1, -0.1, 0.05, -0.05, 0.2, -0.2], [0.5, 2.0]], [[0.1, -0.1, 0.05, -0.05, 0.2, -0.2], [0.5, 2.0]],
    #                            [[0.1, -0.1, 0.05, -0.05, 0.2, -0.2], [0.5, 2.0]], initial=[(0.5, 0.5), (0.2, 0.2, 0.2, 0.2, 0.2), (0.2, 0.2, 0.2, 0.2, 0.2)], maxRandomWalk=0)
    #
    # dict_ = dict()
    # dict_[((0.5, 0.5), (1.5, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0, 1.0))] = 0.5
    # neighbs = getNeighbours(((0.5, 0.5), (1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0, 1.0)), [[0.05, 0.1], [0.5]], [[0.5, 1.0], [0.5, 2.0]], [[0.5, 1.0], [0.5, 2.0]], dict_)


def testHeapq():
    wachtrij = [(-2, ('bla', 3)), (-1, ('blah', 5))]
    heapq.heapify(wachtrij)
    heapq.heappush(wachtrij, (-3, ('ok', 4)))
    print(wachtrij)
    print(heapq.heappop(wachtrij))
    a = (2/10) + (1/10)
    b = round(a, 3)
    print(a)
    print(b)
    c = (3/37) + (4/17)
    print(c)
    c = round(c, 3)
    print(c)
    d = updateTuple((0.125, 0.125, 0.125, 0.125, 0.125), 1.0, 2)
    print(d)



if __name__ == '__main__':
    # kortTestje()
    hillClimbMain()
    # testHeapq()
    # mainCprofile()
    # main()
    # testPDtoNP()