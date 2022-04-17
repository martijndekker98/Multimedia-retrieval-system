import pandas
import sklearn.metrics
import vedo
import evaluation
import queryHelpFunctions
from distanceFunctions import findMostSimilar
import staticVariables as sV
import step4
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import auc
import os


#
# Mainly contains functions used for doing the evaluation
#
#

# Given the database, compute the distance matrix and then find the most similar shapes for each model
# Return a dictionary going from a model (file name) to the results (list of file names most similar to model)
def getResultaten(database: np.ndarray):
    dict_ = dict()
    distMat = queryHelpFunctions.compDistanceMatrix(database)
    for i in range(0, 380):
        model = database[i,1]
        resultaat = queryHelpFunctions.findMostSimilarN(database, i, distMat)
        dict_[model] = [r[0] for r in resultaat]
    return dict_


# given the dataframe, return a dictionary: key: name of file, value: ID (row index)
def getIndexDict(dataframe: pandas.DataFrame):
    indexDict = dict()
    for modelTup in dataframe.iterrows():
        indexDict[modelTup[1].iloc[1]] = modelTup[1].iloc[0]
    return indexDict


# finds the metrics (precision, recall and F1) given a database, dictionary with distances, dictionary going from file name to ID
# the list of categories/classes and the querysize
def findMetricsForQSize(dataframe: pandas.DataFrame, dict_: dict, indexDict: dict, category_list: list, querySize: int,
                        printt: bool = True):
    print(f"Query size: {querySize}")
    conf_matrices = []
    total_conf_mat = np.zeros((2,2), dtype=np.int32)
    for cat in category_list:
        models = dataframe.loc[dataframe['Subfolder'] == cat]
        catIndex = category_list.index(cat)

        conf_mat = np.zeros((2,2), dtype=np.int32)
        for modelTuple in models.iterrows():
            resultaat = dict_[modelTuple[1].iloc[1]]
            tp = 0
            for r in resultaat[:querySize]:
                conf_mat[0 if category_list.index(indexDict[r]) == catIndex else 1, 0] += 1
                if category_list.index(indexDict[r]) == catIndex: tp += 1
            conf_mat[0,1] += 19-tp
            conf_mat[1,1] += sV.maxModelsQuery - 19 - (querySize - tp)
        if printt: print(conf_mat)
        conf_matrices.append(conf_mat)
        total_conf_mat += conf_mat
    precision = total_conf_mat[0,0] / (total_conf_mat[0,0] + total_conf_mat[1,0])
    recall = total_conf_mat[0,0] / (total_conf_mat[0,0] + total_conf_mat[0,1])
    specificity = total_conf_mat[1,1] / (total_conf_mat[1,1] + total_conf_mat[1,0])
    if printt:
        print(total_conf_mat)
        print(f"Precision: {precision} & recall: {recall}")
        tp = sum([x[0,0] for x in conf_matrices])
        fp = sum([x[1,0] for x in conf_matrices])
        fn = sum([x[0,1] for x in conf_matrices])
        tn = sum([x[1,1] for x in conf_matrices])
        print(f"Tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}")
    return precision, recall, specificity


# returns the list of classes/categories from the database
def getCategoryList(dataframe):
    return list(dataframe.iloc[:, 0].unique())


# given a dictionary, pickle the dictionary
def pickleDict(dict_: dict, pickleName: str = 'distanceDict.pickle'):
    print("Pickle dict")
    outfile = open(pickleName, 'wb')
    pickle.dump(dict_, outfile)
    outfile.close()


# given the name of the pickled file, upickle the dictionary in it
def unpickleDict(fileName: str):
    if os.path.isfile(fileName):
        infile = open(fileName, 'rb')
        dict_ = pickle.load(infile)
        infile.close()
        return dict_
    else:
        print("Error, file does not exist")


#
#
# Below for testing only
#
#


def findRoc2(dataframe: pandas.DataFrame, dict_: dict):
    category_list = getCategoryList(dataframe)
    indexDict = getIndexDict(dataframe)
    recalls = []
    precisions = []
    specificities = []
    for querySize in range(1, 380):
        prec, rec, spec = findMetricsForQSize(dataframe, dict_, indexDict, category_list, querySize, printt=False)
        precisions.append(prec)
        recalls.append(rec)
        specificities.append(spec)

    min_prec, max_prec, min_rec, max_rec, min_spec, max_spec = min(precisions), max(precisions), min(recalls), max(recalls), min(specificities), max(specificities)
    print(f"Precision min & max: {min_prec} & {max_prec}")
    print(f"Recall min & max: {min_rec} & {max_rec}")
    print(f"Specificity min & max: {min_spec} & {max_spec}")
    plt.plot(recalls, specificities)
    plt.xlabel("Recall")
    plt.ylabel("Specificity")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    print("Start calculating AUC")
    auc_ = auc(recalls, specificities)
    print(auc_)
    print("Done")
    plt.show()



def findRoc(count: int):
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    category_list = list(dataframe.iloc[:, 0].unique())
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # dataframe.loc[dataframe['File_name'] == name_knn]
    for id, klasse in enumerate(category_list):
        models = dataframe.loc[dataframe['Subfolder'] == klasse]
        klasseIndex = int((int(models.iloc[0, 1].split('.')[0])-1)/20)
        voorspelling = []
        for modelTuple in models.iterrows():
            model = modelTuple[1]
            resultaat = findMostSimilar(dataframe, model.iloc[1], count)
            # print(resultaat)
            # ('xx.ply', *score*) => int(xx) => xx/20 => int value corresponding to the class
            classes = [int((int(r[0].split('.')[0])-1)/20) for r in resultaat]
            voorspelling.extend(classes)
        print(voorspelling.count(klasseIndex))
        binaryPrediction = [1 if x == klasseIndex else 0 for x in voorspelling]
        print(binaryPrediction)
        correct = [1]*200
        correct2 = [0]*200
        correct.extend(correct2)
        a = [1]*100
        b = [0]*100
        binaryPrediction.extend(a)
        binaryPrediction.extend(b)
        roc_values = sklearn.metrics.roc_curve(correct, binaryPrediction)
        print(roc_values)
        break



def main():
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    # dict_ = unpickleDict('distanceDict.pickle')
    database = dataframe.to_numpy(copy=True)
    dict_ = getResultaten(database)
    # pickleDict(dict_)
    findRoc2(dataframe, dict_)

#AUC 0.7679081417764171
#AUC 0.8052725590667574


if __name__ == '__main__':
    main()
    # testje1()