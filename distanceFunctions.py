import math

import numpy as np
import pandas.core.frame
import vedo
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean, cosine

import staticVariables
import staticVariables as sV
import step4


# Computes the distance of the global features using of the possible distance metrics (angular distance, cosine distance,
# Euclidean distance) with or without using globalWeights.
# Expects two 'models', lists containing the global feature values for 2 models
def compDistanceGlobalpart(model1, model2, useEuclid: bool = True, useAngular: bool = False, globalWeights: list = sV.weightsGlobalFeatures, useWeightedVersion: bool = False):
    if useWeightedVersion:
        # print("UseWeighted")
        if not useAngular: # FFT
            distance = cosine(model1, model2, globalWeights)
            return distance
        elif useAngular: # FTT
            cos_sim = 1 - cosine(model1, model2, globalWeights)
            if cos_sim < 1.0 and cos_sim > -1.0:
                return math.acos(cos_sim) / math.pi
            elif cos_sim >= 1.0:
                return math.acos(1.0) / math.pi
            else:
                return math.acos(-1.0) / math.pi
    else:
        if useEuclid: # T?F
            distance = euclidean(model1, model2, w=globalWeights) if len(globalWeights) == len(model1) else euclidean(model1, model2)
            # print(distance)
            return distance
        elif not useAngular: # FFF
            distance = 1 - cosine_similarity([model1], [model2])[0][0]
            # print(distance)
            return distance
        else: # FTF
            cos_sim = cosine_similarity([model1], [model2])[0][0]
            if cos_sim < 1.0 and cos_sim > -1.0:
                return math.acos(cos_sim) / math.pi
            elif cos_sim >= 1.0:
                return math.acos(1.0) / math.pi
            else:
                return math.acos(-1.0) / math.pi


# compute the distance, between 2 models, for the global features given two rows from the database csv file
def compDistanceGlobal(feats1: list, feats2: list, useEuclid: bool = True, useAngular: bool = False, globalWeights: list = sV.weightsGlobalFeatures, useWeighted: bool = False):
    model1 = feats1.iloc[sV.startGlobal:sV.endGlobal] if type(feats1) == pandas.core.frame.DataFrame else feats1[sV.startGlobal:sV.endGlobal]
    model2 = feats2.iloc[sV.startGlobal:sV.endGlobal] if type(feats2) == pandas.core.frame.DataFrame else feats2[sV.startGlobal:sV.endGlobal]
    return compDistanceGlobalpart(model1, model2, useEuclid, useAngular, globalWeights, useWeighted)


# Compute the distance, between 2 models, for the shape property features given two rows from the database csv file
def compDistanceShape(feats1: list, feats2: list):
    # print(f"{len(feats1[sV.endGlobal:])} vs {sum(sV.binCount)}")
    assert len(feats1[sV.endGlobal:]) == sum(sV.binCount)
    assert len(feats2[sV.endGlobal:]) == sum(sV.binCount)
    ans = []
    for f_id, feat in enumerate(sV.featureLabels):
        start = sV.endGlobal+sum(sV.binCount[:f_id])
        model1 = feats1.iloc[start:start+sV.binCount[f_id]] if type(feats1) == pandas.core.frame.DataFrame else feats1[start:start+sV.binCount[f_id]]
        model2 = feats2.iloc[start:start+sV.binCount[f_id]] if type(feats2) == pandas.core.frame.DataFrame else feats2[start:start+sV.binCount[f_id]]

        emd_ = wasserstein_distance(model1, model2)
        emd = (emd_ - sV.histDistAvgs[f_id])/sV.histDistStds[f_id]
        ans.append(emd)
    return ans


# compute the total distance between two models (row from csv database)
def compDistance(feats1, feats2):
    globalDist = compDistanceGlobal(feats1, feats2, sV.useEuclidean, sV.useAngular, sV.weightsGlobalFeatures, sV.useWeightedVersion)
    shapeDistances = compDistanceShape(feats1, feats2)
    distance = sV.weightsGlobalAndShape[0] * globalDist
    som = sum(sV.weightsShapeFeatures)
    for i, d in enumerate(shapeDistances):
        distance += sV.weightsGlobalAndShape[1] * d * (sV.weightsShapeFeatures[i]/som)
    return distance


# Given the table/database/csv file (dataframe), and two names (of models): retrieve their feature values (Rows)
# and then compute the distance between them.
def compareFeatures(dataframe: pandas.core.frame.DataFrame, m1_name: str, m2_name: str):
    resultaat1 = dataframe.loc[dataframe['File_name'] == m1_name]  # .iloc[0,:]
    resultaat2 = dataframe.loc[dataframe['File_name'] == m2_name]  # .iloc[0,:]
    if resultaat1.empty or resultaat2.empty:
        print(f"Error: file name is wrong, because a model is empty, m1 empty: {resultaat1.empty}, m2 empty: {resultaat2.empty}")
        return -1
    else:
        rij1 = resultaat1.iloc[0,:]
        rij2 = resultaat2.iloc[0,:]
        distance = compDistance(rij1, rij2)
        return distance


# Given the database/csv file, and a queried model (m1_name), and the query size: find the most similar models
def findMostSimilar(dataframe: pandas.core.frame.DataFrame, m1_name: str, countt: int = 5):
    """
    Query the shapes in the database most similar to model1...
    :param dataframe: the dataframe that you loaded in, use "dataframe = step4.readCSVAsDataFrame('testDB/features.csv')"
    :param m1_name: the name of the model that you want to query, e.g. '62.ply' (will not work if you use the wrong name)
    :param countt: the number of models to be returned, e.g. the 5 most similar

    :return: array of tuples, with the name of the model as first element and distance as second
    """
    ans = []
    resultaat1 = dataframe.loc[dataframe['File_name'] == m1_name]  # .iloc[0,:]
    if resultaat1.empty:
        print(f"Error: file name is wrong, because m1 empty: {resultaat1.empty}")
        return -1
    else:
        model1 = resultaat1.iloc[0,:]
        modelsNegated = dataframe.loc[dataframe['File_name'] != m1_name]
        for rij in range(0, modelsNegated.shape[0]):
            rij = modelsNegated.iloc[rij,:]
            dist = compDistance(model1, rij)
            ans.append((rij.iloc[1], dist))
    sortedAns = sorted(ans, key=lambda tup: tup[1])
    return sortedAns[:countt]


# Given the features file name, load the csv file. Then for each model and each shape distance:
# compute the distances between the models and same them in 5 tables
def calcHistDists(featureFile: str = 'testDB/features.csv'):
    dataframe = step4.readCSVAsDataFrame(featureFile)
    features = [[], [], [], [], []]
    featureLabels = ["A3", "D1", "D2", "D3", "D4"]

    for f_id, feature in enumerate(featureLabels):
        print(f"Feature {feature}")
        tabel = np.zeros((dataframe.shape[0], dataframe.shape[0]))
        # print(f"bins: {staticVariables.binCount[:f_id]}, sum: {sum(staticVariables.binCount[:f_id])}")
        startIndex = staticVariables.endGlobal + sum(staticVariables.binCount[:f_id])
        print(f"start: {startIndex} <-> {startIndex+staticVariables.binCount[f_id]}")
        for m1 in range(0, dataframe.shape[0]):
            model1 = dataframe.iloc[m1, startIndex:startIndex + staticVariables.binCount[f_id]]
            for m2 in range(m1+1, dataframe.shape[0]):
                model2 = dataframe.iloc[m2, startIndex:startIndex + staticVariables.binCount[f_id]]
                distance = wasserstein_distance(model1, model2)
                tabel[m1, m2] = distance
                tabel[m2, m1] = distance
        features[f_id] = tabel
    print("Done!")
    return features


# Given the feature file name, compute the distances for the shape features: gives 5 tables of m*m size
# (m = models in database). Then save these 5 tables in 5 separate csv files
def makeHistCSV(features: str, fileNameStart: str):
    features = calcHistDists(features)
    for f_id, feat in enumerate(features):
        step4.writeToCsv(f"{fileNameStart}_{sV.featureLabels[f_id]}.csv", [], feat)


#
# Below only for testing
#

# For testing
def main():
    print("find most similar")
    dataframe = step4.readCSVAsDataFrame('testDB/featuresComb.csv')
    resultaten = findMostSimilar(dataframe, '5.ply', 5)
    print(resultaten)
    print(euclidean([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]))


# For testing
def main2():
    print("Main")
    # model1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # model2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # model3 = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    # compDistanceGlobal(model2, model2, False, True)
    # compDistanceGlobal(model2, model1, False, True)
    # compDistanceGlobal(model2, model3, False, True)
    # dataframe = step4.readCSVAsDataFrame('testDB/test.csv')
    dataframe = step4.readCSVAsDataFrame('testDB/features.csv')
    afstand = compareFeatures(dataframe, '61.ply', '62.ply')
    print(f"Distance: {afstand}")

    negated = dataframe.loc[dataframe['File_name'] != '61.ply']
    print(negated.shape[0])
    namen = negated.iloc[:,1]
    naamr = negated.iloc[0,:]
    naam = naamr.iloc[1]
    print(naam)
    print(type(naam))
    lijst = [('a', 0.1),('b', 0.3),('c', 0.081),('d', 0.000001),('e', 0.000002)]
    sortedd = sorted(lijst, key=lambda tup: tup[1])
    print(sortedd)


if __name__ == '__main__':
    pass

