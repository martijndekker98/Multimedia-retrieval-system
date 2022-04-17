import cProfile
import pickle
import pstats
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas
import vedo
from sklearn.metrics import auc

import compRandomSystem
import queryHelpFunctions as qHF
import roc
import staticVariables as sV
import step4
import testParametersNP


#
# This file contains functions to compute the precision, recall, F1 etc.
# Also contains methods to plot the results in graphs
#

# Plot a bar graph with vertical bars
def plotMultipleBarGraph(values: list, labels:list, xPoints: list, xlabel: str, ylabel: str, barWidth: float = -1,
                         ylim: list = [], horizGridLines: bool = True, addText: bool = True, yticksStep: float = -1):
    assert len(values) > 0
    assert len(values[0]) == len(xPoints)
    assert len(labels) == len(values)
    fig,ax = plt.subplots(figsize=(12, 8))
    numbOfBars = len(values)
    if barWidth == -1: barWidth = 1/(numbOfBars+1)
    print(barWidth)

    ylim_ = [0, 1.1*max([max(x) for x in values])]
    # Make the plot
    if horizGridLines:
        ax.grid(axis='y', zorder=0)
    bars = [np.arange(len(values[0]))]
    print(bars[0])
    for p_id, part in enumerate(values):
        if p_id == 0: br = bars[0]
        else:
            br = [x + barWidth for x in bars[p_id-1]]
            bars.append(br)
        plt.bar(br, values[p_id], width=barWidth, label=labels[p_id], zorder=3)
        if addText:
            for v_id, v in enumerate(values[p_id]):
                plt.text(br[v_id] - 0.4*barWidth, v + (1.08 * ylim_[1] / 100), "{:.4f}".format(v)[1:], fontsize=10, rotation=0, zorder=3)
                #plt.text(x + (0.08 * binsDiff), y + (1.08 * max(density) / 100), "{:.4f}".format(num / sum(count))[1:],
                 #        fontsize=10, rotation=0)  # x,y,str

    # Adding Xticks
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim_)
    xticksV = (numbOfBars-1)/2 * barWidth
    plt.xticks([r + xticksV for r in range(len(values[0]))], xPoints)
    if yticksStep != -1:
        yticks_ = np.arange(0.0, ylim_[1], yticksStep)
        plt.yticks(yticks_)

    plt.tight_layout()
    plt.legend()
    plt.show()


# Plot a horizontal bar graph with 1 bar per y value
def plotHBarGraph(valueLabelTuples: list, xlabel: str, ylabel: str, barWidth: float = -1,
                          xlim: list = [], vertGridLines: bool = True, addText: bool = True, xticksStep: float = -1):
    fig,ax = plt.subplots(figsize=(8, 10))
    # fig, ax = plt.subplots()
    numbOfBars = len(valueLabelTuples)
    if barWidth == -1: barWidth = 1/(numbOfBars+1)
    print(barWidth)

    xlim_ = [0, 1.1*max([x[0] for x in valueLabelTuples])]
    # Make the plot
    if vertGridLines:
        ax.grid(axis='x', zorder=0)
    plt.barh([x[1] for x in valueLabelTuples], [x[0] for x in valueLabelTuples], zorder=3)
    # for p_id, part in enumerate(valueLabelTuples):
    #     # plt.barh(part[1], part[0], zorder=3)
    #     # plt.bar(br, values[p_id], width=barWidth, label=labels[p_id], zorder=3)
    #     if addText:
    #         plt.text(part[1], part[0] + 0.02, "{:.4f}".format(part[0])[1:], fontsize=10, rotation=0, zorder=3)
    #         # for v_id, v in enumerate(values[p_id]):
    #         #     plt.text(br[v_id] - 0.4*barWidth, v + (1.08 * ylim_[1] / 100), "{:.4f}".format(v)[1:], fontsize=10, rotation=0, zorder=3)
    #             #plt.text(x + (0.08 * binsDiff), y + (1.08 * max(density) / 100), "{:.4f}".format(num / sum(count))[1:],
    #              #        fontsize=10, rotation=0)  # x,y,str
    for i, v in enumerate(valueLabelTuples):
        plt.text(v[0]+0.005, i-0.2, "{:.4f}".format(v[0])[1:],fontsize=10, rotation=0, zorder=3)

    # Adding Xticks
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim_)
    # xticksV = (numbOfBars-1)/2 * barWidth
    # plt.xticks([r + xticksV for r in range(len(values[0]))], xPoints)
    if xticksStep != -1:
        xticks_ = np.arange(0.0, xlim_[1], xticksStep)
        plt.xticks(xticks_)

    plt.tight_layout()
    # plt.legend()
    plt.show()


# Plot a graph with multiple horizontal bars per y value
def plotMultipleHBarGraph2(valueLabelTuples: list, barLabels: list, xlabel: str, ylabel: str, barWidth: float = -1,
                           xlim: list = [], vertGridLines: bool = True, addText: bool = True, xticksStep: float = -1,
                           boldYLAbes: list = [], barColours: list = [], figSize: tuple = None, textAfterBar: bool = True):
    """
    :param valueLabelTuples: tuple of (values: list, class name:str)
    """
    fig,ax = plt.subplots(figsize=(8, 10)) if figSize is None else plt.subplots(figsize=figSize)
    # fig, ax = plt.subplots()
    if barWidth == -1: barWidth = 0.8/len(barLabels)
    print(f"Barwidth: {barWidth}")

    xlim_ = [0, 1.1*max([max(x[0]) for x in valueLabelTuples])]
    if vertGridLines: ax.grid(axis='x', zorder=0)

    # Make the plot
    ind = np.arange(len(valueLabelTuples))
    print(f"IND: {ind}")
    for b_id, barl in enumerate(barLabels):
        if not barColours:
            ax.barh(ind + (b_id*barWidth), [tpl[0][b_id] for tpl in valueLabelTuples], barWidth, label=barl, zorder=3)
        else:
            ax.barh(ind + (b_id*barWidth), [tpl[0][b_id] for tpl in valueLabelTuples], barWidth, label=barl, zorder=3, color=barColours[b_id])

    plt.yticks(ind + ((len(barLabels)-1) * 0.5*barWidth), [x[1] for x in valueLabelTuples])
    # print(matplotlib.axes.Axes.get_yticklabels(ax))
    plt.ylim([-1.5*barWidth, len(valueLabelTuples)])
    # ax.set(yticks=ind + ((len(barLabels)-1) * 0.5*barWidth), yticklabels=[x[1] for x in valuesTpls], ylim=[-1.5*barWidth, len(valueLabelTuples)]) #2 * barWidth - 1.2, len(valuesTpls)

    fontSizes = [10, 10, 10, 7]
    if addText:
        for v_id, v in enumerate(valueLabelTuples):
            if v[1] in boldYLAbes:
                ax.get_yticklabels()[v_id].set_weight('bold')
                # if barColours:
                #     ax.containers[1][v_id].set_color(barColours[len(barLabels)+1])
                #     ax.containers[0][v_id].set_color(barColours[len(barLabels)])
            for s_id, subbar in enumerate(v[0]):
                if textAfterBar: #0.7 > 0.005
                    plt.text(subbar + (xlim_[1]*0.0071), ind[v_id] + (s_id*barWidth)-(0.3*barWidth), "{:.4f}".format(subbar)[1:],
                             fontsize=fontSizes[len(v[0])], rotation=0, zorder=3)
                else: # 0.7 max -> 0.035
                    plt.text(subbar - (xlim_[1]*0.06), ind[v_id] + (s_id*barWidth)-(0.3*barWidth), "{:.4f}".format(subbar)[1:],
                             fontsize=fontSizes[len(v[0])], rotation=0, zorder=3, color='white')

    # Adding Xticks
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim_)
    if xticksStep != -1:
        xticks_ = np.arange(0.0, xlim_[1], xticksStep)
        plt.xticks(xticks_)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    plt.tight_layout()
    # plt.legend()
    plt.show()


# Computes the F1 for different distance functions and weights (euclidean, cosine, angular)
def compareDistMetrics(querySizes: list, dataframe: pandas.DataFrame):
    database = dataframe.to_numpy(copy=True)
    category_list = roc.getCategoryList(dataframe)
    indexDict = roc.getIndexDict(dataframe)

    f1s = []
    for mode in range(0, 5):
        if mode < 3: sV.useWeightedVersion = False
        else: sV.useWeightedVersion = True
        if mode == 0:
            sV.useEuclidean = True
            sV.useAngular = False
        elif mode == 1:
            sV.useEuclidean = False
            sV.useAngular = True
        elif mode == 2:
            sV.useEuclidean = False
            sV.useAngular = False
        elif mode == 3:
            sV.useEuclidean = False
            sV.useAngular = True
        else:
            sV.useEuclidean = False
            sV.useAngular = False

        dict_ = roc.getResultaten(database)
        f1_eu = []
        for q in querySizes:
            prec, rec, _ = roc.findMetricsForQSize(dataframe, dict_, indexDict, category_list, q, printt=False)
            f1_eu.append(((2 * prec * rec) / (prec + rec)))
            # f1_eu.append(prec)
        f1s.append(f1_eu)
    plotMultipleBarGraph(f1s, ['Euclidean', 'Angular distance', 'Cosine distance', 'Angular distance weighted', 'Cosine distance weighted'], querySizes, 'Query size', 'F1', yticksStep=0.05)


# Get (precision, recall or f1) for the querySizes and a system (mods)
def getMetricsPerClass(querySizes: list, modus: int):
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    database = testParametersNP.getDatabase(dataframe)
    distMat = qHF.compDistanceMatrix(database)

    print(f"Querysizes: {querySizes}")
    precisions = np.zeros((len(querySizes), 19))
    recalls = np.zeros((len(querySizes), 19))
    f1s = np.zeros((len(querySizes), 19))
    # confMatTotal = np.array([np.zeros((2,2), dtype=np.int32) for q in querySizes])
    # confMats = []
    # for q in querySizes:
    #     a = []
    #     for klasse in range(0, 19):
    #         a.append(np.zeros((2,2), dtype=np.int32))
    #     confMats.append(a)
    # confMats = np.array([np.array(xi) for xi in confMats])
    #
    # for i in range(0, len(database)):
    #     catIndex = database[i, 0]
    #     mostSimilar = qHF.findMostSimilarN(database, i, distMat)
    #     for qID, qSize in enumerate(querySizes):
    #         confMat = testParametersNP.computeConfMat(mostSimilar, qSize, catIndex)
    #         confMatTotal[qID] += confMat
    #         confMats[qID, catIndex] += confMat
    # confMatTotal, confMats = getConfMatricesNormal(database, distMat, len(querySizes))
    confMatTotal_, confMats_ = getConfMatrices(dataframe, database, modus, distMat, 379)
    confMatTotal = np.array([confMatTotal_[x-1] for x in querySizes])
    confMats = np.array([confMats_[x-1] for x in querySizes])

    # Compute the metrics
    totalPrecisions = [0]*len(querySizes)
    totalRecalls = [0]*len(querySizes)
    totalF1s = [0]*len(querySizes)
    for qID, qSize in enumerate(querySizes):
        precision_ = confMatTotal[qID,0, 0] / (confMatTotal[qID,0, 0] + confMatTotal[qID,1, 0])
        recall_ = confMatTotal[qID,0, 0] / (confMatTotal[qID,0, 0] + confMatTotal[qID,0, 1])
        totalPrecisions[qID] += precision_
        totalRecalls[qID] += recall_
        totalF1s[qID] += (2 * precision_ * recall_) / (precision_ + recall_)
        for cl_id in range(0, confMats[qID].shape[0]):
            precision = confMats[qID, cl_id,0, 0] / (confMats[qID, cl_id, 0, 0] + confMats[qID,cl_id,1, 0])
            recall = confMats[qID,cl_id,0, 0] / (confMats[qID,cl_id,0, 0] + confMats[qID,cl_id, 0, 1])
            precisions[qID, cl_id] = precision
            recalls[qID, cl_id] = recall
            f1s[qID, cl_id] = (2 * precision * recall) / (precision + recall)
    return totalPrecisions, totalRecalls, totalF1s, precisions, recalls, f1s


# Given a query size and a modus (0: precision, 1: recall, 2: f1) compute the results for selected weights, equal weights
# and knn. Next plot these results in a horizontal bar graph
def plotMetric(querySize: int, modus: int):
    """Modus (int), 0 -> precision, 1 -> recall, 2 -> f1"""
    sV.weightsGlobalFeatures = [0.47, 0.39, 0.06, 0.01, 0.08]
    sV.weightsShapeFeatures = [0.16, 0.21, 0.31, 0.21, 0.11]
    sV.weightsGlobalAndShape = [0.65, 0.35]
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    if modus < 0 or modus > 2:
        print("Incorrect modus")
        return
    category_list = roc.getCategoryList(dataframe)
    tp1, tr1, tf1, ps1, rs1, f11 = getMetricsPerClass([querySize], 0)
    print(ps1[0])
    print(ps1[0,0])

    tp3, tr3, tf3, ps3, rs3, f13 = getMetricsPerClass([querySize], 1)
    # Change the weights
    sV.weightsGlobalFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
    sV.weightsShapeFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
    sV.weightsGlobalAndShape = [0.5, 0.5]
    tp2, tr2, tf2, ps2, rs2, f12 = getMetricsPerClass([querySize], 0)
    tuples_ = []
    if modus == 0: tuples_.append( ([tp1[0], tp2[0], tp3[0]], 'Average') )
    elif modus == 1: tuples_.append( ([tr1[0], tr2[0], tr3[0]], 'Average') )
    elif modus == 2: tuples_.append( ([tf1[0], tf2[0], tf3[0]], 'Average') )
    for i in range(0, len(category_list)):
        if modus == 0: tuples_.append( ([ps1[0,i], ps2[0,i], ps3[0,i]], category_list[i]) )
        elif modus == 1: tuples_.append( ([rs1[0,i], rs2[0,i], rs3[0,i]], category_list[i]) )
        elif modus == 2: tuples_.append( ([f11[0,i], f12[0,i], f13[0,i]], category_list[i]) )
    tps = sorted(tuples_, key = lambda tup: tup[0][0])
    xas = 'Precision' if modus == 0 else 'Recall' if modus == 1 else 'F1'
    plotMultipleHBarGraph2(tps, ['Selected weights', 'Equal weights', 'KNN'], xas, 'Class', xticksStep=0.1,
                           boldYLAbes=['Average'], barColours=['blue', 'dodgerblue', 'mediumblue', 'green', 'lime'],
                           textAfterBar=True, figSize=(6,11))


#
# Getting the confusion matrices
def getConfMatrices(dataframe: pandas.DataFrame, database: np.ndarray, modus: int, distMat, querysizes: int):
    if modus == 0:
        confTotal, confmats = getConfMatricesNormal(database, distMat, querysizes)
    elif modus == 1:
        confTotal, confmats = getKNNConfMatrices(database, dataframe, querysizes)
    return confTotal, confmats

# Get the confusion matrix for the custom distance function given the database, matrix with distances and querysizes
def getConfMatricesNormal(database: np.ndarray, distMat, querysizes: int):
    print(f"Querysizes: {querysizes}")
    confMatTotal = np.array([np.zeros((2, 2), dtype=np.int32) for q in range(0, querysizes)])
    confMats = []
    for q in range(0, querysizes):
        a = []
        for klasse in range(0, 19):
            a.append(np.zeros((2, 2), dtype=np.int32))
        confMats.append(a)
    confMats = np.array([np.array(xi) for xi in confMats])

    for i in range(0, len(database)):
        catIndex = database[i, 0]
        mostSimilar = qHF.findMostSimilarN(database, i, distMat)
        for qsize in range(1, 380):
            confMat = testParametersNP.computeConfMat(mostSimilar, qsize, catIndex)
            confMatTotal[qsize - 1] += confMat
            confMats[qsize - 1, catIndex] += confMat
    return confMatTotal, confMats


# Compute the confusion matrix for KNN igve the database and the querysizes
def getKNNConfMatrices(database: np.ndarray, dataframe: pandas.DataFrame, querysizes: int):
    with open('knn_tree.pickle', 'rb') as handle:
        tree = pickle.load(handle)

    confMatTotal = np.array([np.zeros((2, 2), dtype=np.int32) for q in range(0, querysizes)])
    confMats = []
    for q in range(0, querysizes):
        a = []
        for klasse in range(0, 19):
            a.append(np.zeros((2, 2), dtype=np.int32))
        confMats.append(a)
    confMats = np.array([np.array(xi) for xi in confMats])

    for i in range(0, len(database)):
        catIndex = database[i, 0]
        query = dataframe.loc[dataframe['File_name'] == dataframe.iloc[i,1]]
        query_np = query.to_numpy()[0][2:].reshape(1, -1)
        dist, ind = tree.query(query_np, k=380)
        mostSimilar = [(0, 0, database[x,0]) for x in ind[0][1:]]
        for qsize in range(1, 380):
            confMat = testParametersNP.computeConfMat(mostSimilar, qsize, catIndex)
            confMatTotal[qsize - 1] += confMat
            confMats[qsize - 1, catIndex] += confMat
    return confMatTotal, confMats


# Compute the roc for a distance function configuration (modus)
def computeROC(dataframe: pandas.DataFrame, modus: int, startWithZeroValues: bool = True):
    database = testParametersNP.getDatabase(dataframe)
    distMat = qHF.compDistanceMatrix(database)
    qSizes = 380 if startWithZeroValues else 379
    startValue = 1 if startWithZeroValues else 0

    specificities = np.zeros((qSizes, 19))
    recalls = np.zeros((qSizes, 19))
    confMatTotal, confMats = getConfMatrices(dataframe, database, modus, distMat, 379)
    # confMatTotal = np.array([np.zeros((2,2), dtype=np.int32) for q in range(1, 380)])
    # confMats = []
    # for q in range(0, 379):
    #     a = []
    #     for klasse in range(0, 19):
    #         a.append(np.zeros((2,2), dtype=np.int32))
    #     confMats.append(a)
    # confMats = np.array([np.array(xi) for xi in confMats])
    #
    # for i in range(0, len(database)):
    #     catIndex = database[i, 0]
    #     mostSimilar = qHF.findMostSimilarN(database, i, distMat)
    #     for qsize in range(1, 380):
    #         confMat = testParametersNP.computeConfMat(mostSimilar, qsize, catIndex)
    #         confMatTotal[qsize-1] += confMat
    #         confMats[qsize-1, catIndex] += confMat

    # Compute the metrics
    totalRecalls = [0]*qSizes
    totalSpecificities = [0]*qSizes
    for qsize_ in range(startValue, qSizes):
        qsize = qsize_ - 1 if startWithZeroValues else qsize_
        recall_ = confMatTotal[qsize,0, 0] / (confMatTotal[qsize,0, 0] + confMatTotal[qsize,0, 1])
        specificity_ = confMatTotal[qsize,1, 1] / (confMatTotal[qsize,1, 1] + confMatTotal[qsize,1, 0])
        totalRecalls[qsize_] += recall_
        totalSpecificities[qsize_] += specificity_
        for cl_id in range(0, confMats[qsize].shape[0]):
            recall = confMats[qsize,cl_id,0, 0] / (confMats[qsize,cl_id,0, 0] + confMats[qsize,cl_id, 0, 1])
            specificity = confMats[qsize,cl_id,1, 1] / (confMats[qsize,cl_id,1, 1] + confMats[qsize,cl_id,1, 0])
            recalls[qsize_, cl_id] = recall
            specificities[qsize_,cl_id] = specificity
            if qsize == 0 and startWithZeroValues: specificities[qsize,cl_id] = 1.0
    if startWithZeroValues: totalSpecificities[0] = 1.0
    return totalRecalls, totalSpecificities, recalls, specificities


# Plot a line graph (used for ROC)
def plotLineGraph(valueLabelPairs: list, xlabel: str, ylabel: str, showGridlines: bool = True, xtickStep: float = 0.1,
                  ytickstep: float = 0.1, xlim:list = None, ylim: list = None, graphSize: tuple = (12,8), updateText: bool = True):
    """
    valueLabelPairs = (x, y, label)
    """
    fig,ax = plt.subplots(figsize=graphSize)
    if showGridlines:
        ax.grid(zorder=0)
    else:
        ax.grid(axis='y', zorder=0)

    for vlp in valueLabelPairs:
        if updateText:
            print(f"AUC of {vlp[2]} = {vlp[3]}")
            aucRounded = "{:.4f}".format(vlp[3])
            labelText = f"{vlp[2]} ({aucRounded})"
        else:
            labelText = vlp[2]
        plt.plot(vlp[0], vlp[1], label=labelText, zorder=3)
    # auc_ = auc(recalls, specificities)
    xlim_ = xlim if xlim is not None else [0.0, 1.01]
    ylim_ = ylim if ylim is not None else [0.0, 1.01]
    yticks_ = np.arange(0.0, ylim_[1], ytickstep)
    xticks_ = np.arange(0.0, xlim_[1], xtickStep)
    plt.xlim(xlim_)
    plt.ylim(ylim_)
    plt.yticks(yticks_)
    plt.xticks(xticks_)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower left')
    # plt.legend(loc='lower left')
    plt.show()


# Compute and plot the roc for the custom distance function using the selected weights on a per class basis
def computeAndPlotROC(perClass: bool, modus: int):
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    category_list = roc.getCategoryList(dataframe)
    tr, ts, rs, ss = computeROC(dataframe, modus)
    if perClass:
        ans = [(rs[:,i], ss[:,i], category_list[i], auc(rs[:,i], ss[:,i])) for i in range(0, 19)]
        ans_ = sorted(ans, key = lambda tup: tup[3])
        plotLineGraph(ans_, 'Recall', 'Specificity')
    else:
        plotLineGraph([(tr, ts, 'Average')], 'Recall', 'Specificity')


# Compute and plot the roc curves for the selected weights, equal weights, random and KNN
def computeAndPlotROCModels():
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    tr, ts, rs, ss = computeROC(dataframe, 0, True)
    tr2, ts2, rs2, ss2 = computeROC(dataframe, 1, True)

    sV.weightsGlobalFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
    sV.weightsShapeFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
    sV.weightsGlobalAndShape = [0.5, 0.5]
    tr3, ts3, rs3, ss3 = computeROC(dataframe, 0, True)

    print(f"Selected weights, the first point: {tr[0]}, {ts[0]} & last: {tr[-1]}, {ts[-1]}")
    print(f"Equal weights, the first point: {tr3[0]}, {ts3[0]} & last: {tr3[-1]}, {ts3[-1]}")
    print(f"KNN, the first point: {tr2[0]}, {ts2[0]} & last: {tr2[-1]}, {ts2[-1]}")
    tr1, ts1 = tr, ts
    # tr1, ts1 = [0.0], [1.0]
    # tr1.extend(tr)
    # ts1.extend(ts)
    tr4, ts4 = compRandomSystem.compROCValues()
    ans = [(tr1, ts1, 'Selected weights', auc(tr, ts)), (tr3, ts3, 'Equal weights', auc(tr3, ts3)), (tr2, ts2, 'KNN', auc(tr2, ts2)), (tr4, ts4, 'Random', auc(tr4, ts4))]
    ans_ = sorted(ans, key = lambda tup: tup[3])
    plotLineGraph(ans_, 'Recall', 'Specificity')


# Make the table containing the AUC value for the selected weights, equal weights, KNN per class and averaged.
# Prints out a overleaf formatted table
def makeROCTable():
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    category_list = roc.getCategoryList(dataframe)
    tr, ts, rs, ss = computeROC(dataframe, 0)
    tr2, ts2, rs2, ss2 = computeROC(dataframe, 1)

    sV.weightsGlobalFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
    sV.weightsShapeFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
    sV.weightsGlobalAndShape = [0.5, 0.5]
    tr3, ts3, rs3, ss3 = computeROC(dataframe, 0)
    AUCs1 = [auc(rs[:,i], ss[:,i]) for i in range(0, 19)]
    AUCs2 = [auc(rs2[:,i], ss2[:,i]) for i in range(0, 19)]
    AUCs3 = [auc(rs3[:,i], ss3[:,i]) for i in range(0, 19)]
    AUC1 = "{:.4f}".format(auc(tr, ts))
    AUC2 = "{:.4f}".format(auc(tr2, ts2))
    AUC3 = "{:.4f}".format(auc(tr3, ts3))
    print('\\textbf{Average} & '+ f'{AUC1} & {AUC3} & {AUC2} \\\\')
    ans = [(category_list[i], AUCs1[i], AUCs3[i], AUCs2[i]) for i in range(0, 19)]
    ans_ = sorted(ans, key=lambda tup: tup[1], reverse=True)
    for i in ans_:
        a = "{:.4f}".format(i[1])
        b = "{:.4f}".format(i[2])
        c = "{:.4f}".format(i[3])
        print(f'{i[0]} & {a} & {b} & {c}\\\\')


# compute and plot the mean average precision for the selected weights, equal weights and KNN given the min and max query sizes
def compAndPlotMAP(minQuerySize: int, maxQuerySize: int):
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    database = testParametersNP.getDatabase(dataframe)
    category_list = roc.getCategoryList(dataframe)
    distMat = qHF.compDistanceMatrix(database)

    minQuerySize -= 1  # array/list starts with index 0, but querying starts with 1
    maxQuerySize -= 1
    querysizes = maxQuerySize - minQuerySize
    MAPs, MAPtotals = [], []
    for i in range(0,3):  # selected weights, knn, equal
        modus = i%2
        if i == 2:
            print(f"Updating the weights")
            sV.weightsGlobalFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
            sV.weightsShapeFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
            sV.weightsGlobalAndShape = [0.5, 0.5]
            distMat = qHF.compDistanceMatrix(database)

        confMatTotal, confMats = getConfMatrices(dataframe, database, modus, distMat, 379)

        precisions = np.zeros((querysizes, 19))
        totalPrecisions = [0]*querysizes  # Total(all classes) for each query size
        querisToCheck = [x for x in range(minQuerySize, maxQuerySize)]
        print(f"Queries, to check indexes: {querisToCheck}")
        for qID, q in enumerate(querisToCheck):
            precision_ = confMatTotal[q,0, 0] / (confMatTotal[q,0, 0] + confMatTotal[q,1, 0])
            totalPrecisions[qID] += precision_
            for cl_id in range(0, confMats[qID].shape[0]): # for each class
                precision = confMats[q, cl_id,0, 0] / (confMats[q, cl_id, 0, 0] + confMats[q,cl_id,1, 0])
                precisions[qID, cl_id] = precision
        map = []
        MAPtotals.append(sum(totalPrecisions)/querysizes)
        for j in range(0, 19):
            som = sum(precisions[:,j])
            map.append(som/querysizes)
        MAPs.append(map)
        modi = ['Selected weights', 'KNN', 'Equal weights']
        print(f"Modus: {modi[i]}, with MAP: {MAPtotals[i]}")

    tuples_ = []
    tuples_.append( ([MAPtotals[0], MAPtotals[2], MAPtotals[1]], 'Average') )
    for i in range(0, len(category_list)):
        tuples_.append( ([MAPs[0][i], MAPs[2][i], MAPs[1][i]], category_list[i]) )
    tps = sorted(tuples_, key = lambda tup: tup[0][0])
    xas = 'Mean average precision'
    plotMultipleHBarGraph2(tps, ['Selected weights', 'Equal weights', 'KNN'], xas, 'Class', xticksStep=0.1,
                           boldYLAbes=['Average'], barColours=['blue', 'dodgerblue', 'mediumblue', 'green', 'lime'],
                           textAfterBar=True, figSize=(6,11))


#
#
# Below only used for testing
#
#


def main():
    # plotMultipleBarGraph([[0.38, 0.5, 0.55, 0.6],[0.3, 0.35, 0.4, 0.34],[0.51, 0.43, 0.47, 0.59]],
    #                      ['Angular distance', 'Cosine Distance', 'c'], ['5', '10', '20', '40'], 'Query size', 'F1', yticksStep=0.05)
    # compareDistMetrics([5, 10, 20, 40])
    #
    valuesTpls = []
    alfbt = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(0, 20):
        # valuesTpls.append(([0.2+(i*0.02), 0.19+(i*0.02)], alfbt[i]))
        valuesTpls.append(([0.2+(i*0.02), 0.19+(i*0.02), 0.185+(i*0.02)], alfbt[i]))
        # valuesTpls.append(([0.2+(i*0.02), 0.19+(i*0.02), 0.185+(i*0.02), 0.195+(i*0.02)], alfbt[i]))
    # plotMultipleHBarGraph(valuesTpls, 'Precision', 'Class')

    # tp, tr, tf, ps, rs, f1 = getMetricsPerClass([5, 10, 20, 40])
    plotMultipleHBarGraph2(valuesTpls, ['w', 'no w', 'x'], 'Precision', 'Class', xticksStep=0.05, boldYLAbes=['c', 'g'],
                           barColours=['blue', 'dodgerblue', 'green', 'lime'], figSize=(6,11), textAfterBar=False)


def computeTime():
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    database = testParametersNP.getDatabase(dataframe)
    # For knn
    with open('knn_tree.pickle', 'rb') as handle:
        tree = pickle.load(handle)

    # indexDict = {i: database[i,1] for i in range(0, dataframe.shape[0])}

    print("Start timing")
    # query = dataframe.loc[dataframe['File_name'] == dataframe.iloc[i,1]]
    # query_np = query.to_numpy()[0][2:].reshape(1, -1)
    times = 10
    timeListKNN = []
    for qs in range(1, 380):
        t1 = time.perf_counter_ns()
        for t in range(0, times):
            for i in range(0, len(database)):
                query_np = database[0,2:].reshape(1, -1)
                dist, ind = tree.query(query_np, k=qs+1)
                # mostSimilar = [(database[x, 1], dist[0][id], database[x,0]) for id, x in enumerate(ind[0][1:])]
        t2 = time.perf_counter_ns()
        timeListKNN.append((t2-t1)/(times*1000000))
    print(timeListKNN)

    print("Start timing 2")
    timeListWeights = []
    t1 = time.perf_counter_ns()
    for t in range(0, times):
        for i in range(0, len(database)):
            mostSimilar = qHF.getQueryResults(database, i)
    t2 = time.perf_counter_ns()
    timeListWeights.append((t2 - t1) / (times*1000000))
    print(timeListWeights)
    timeListWeights_ = [timeListWeights[0] for x in range(1, 380)]
    querySizes = [x for x in range(1, 380)]
    # ans = [(tr1, ts1, 'Selected weights', auc(tr, ts)), (tr3, ts3, 'Equal weights', auc(tr3, ts3)),
    #        (tr2, ts2, 'KNN', auc(tr2, ts2)), (tr4, ts4, 'Random', auc(tr4, ts4))]
    # ans_ = sorted(ans, key=lambda tup: tup[3])
    # plotLineGraph(ans_, 'Recall', 'Specificity')
    ans = [(querySizes, timeListKNN, 'KNN'), (querySizes, timeListWeights_, 'Selected weights')]
    plotLineGraph(ans, 'Query size', 'Time in ms', False, 20, 0.1, updateText=False)
    # Start timing
    # [40.1285591, 40.412807, 40.7326537, 40.8914024, 41.079094, 41.4573985, 41.6535229, 41.8245897, 42.2852351, 42.3601069, 44.7112811, 46.9154523, 47.4514214, 47.7318745, 47.9908841, 48.3307588, 49.296774, 51.7143571, 49.2665964, 49.2929793, 49.5614155, 49.8668059, 50.0380834, 50.3234156, 50.5160464, 50.951574, 51.1463732, 53.4028885, 53.7834818, 53.9518682, 56.0059343, 56.2949107, 56.4377574, 56.4805543, 56.9319537, 57.2418315, 57.4278666, 57.7934616, 57.9507345, 58.3861718, 58.3926772, 58.9089795, 61.4867268, 62.589746, 61.4131199, 60.2490528, 61.0842681, 61.0003611, 61.2222961, 61.5086824, 61.6506859, 62.0162808, 62.4327743, 62.9719056, 63.3241285, 63.6156949, 63.7790817, 64.0356819, 64.7001016, 64.9990167, 65.206676, 65.7395431, 65.8765771, 66.1819675, 66.6555935, 66.8408456, 67.0568173, 67.4513853, 67.7133764, 68.1326406, 68.5013677, 68.7312236, 69.3168563, 69.7054008, 70.0295542, 70.2312503, 70.4711654, 70.8245027, 74.8251159, 71.8018119, 71.4734421, 71.719471, 71.8634321, 73.147246, 74.0218851, 73.9249975, 73.3590315, 73.564914, 74.0836559, 74.3914857, 75.0025977, 75.2843759, 75.3084096, 75.8246517, 76.0301728, 76.0150237, 76.694201, 77.0394969, 77.1042191, 77.5026722, 77.7931242, 77.919015, 78.6722809, 78.8988239, 78.7404064, 78.957312, 79.4674703, 80.3280446, 81.2089782, 82.3402175, 80.9237666, 81.1407925, 81.9267659, 81.7995199, 81.9942891, 82.2239945, 82.5240239, 82.7988451, 82.9437699, 83.1932022, 83.6845673, 83.696554, 84.1653011, 84.2204762, 84.5534841, 85.0307544, 85.8363041, 86.1333519, 86.2980338, 86.6041771, 86.9058028, 87.2528454, 87.6003398, 87.714545, 88.1451936, 87.9761747, 88.5790345, 88.8521088, 89.0222722, 89.4880074, 89.6753677, 90.1160453, 90.0054241, 89.9596759, 91.004689, 90.7395054, 91.1729247, 91.3499848, 91.7345839, 92.1831222, 93.0066219, 92.5215211, 92.6381357, 92.7897165, 93.4360658, 93.9590542, 93.3456835, 93.9316775, 94.2441153, 94.1661112, 95.0569535, 94.9019092, 95.4234821, 95.6351773, 95.8903618, 96.3477246, 96.9151963, 96.7687054, 96.7655732, 97.0615368, 98.3598672, 97.7281551, 97.7805894, 98.3420377, 98.4382026, 98.779794, 99.0580184, 99.2666415, 99.6165455, 99.7780348, 100.0772511, 100.0287923, 100.8237407, 100.9919764, 101.1121147, 101.3206475, 101.9626298, 101.9479926, 103.2852348, 102.4142701, 102.6163879, 106.0880494, 103.0926643, 103.4434415, 103.3655579, 104.6560278, 103.9663998, 110.5155465, 105.3873985, 104.8197158, 104.8679036, 105.6680624, 105.6444504, 105.6783926, 106.0745267, 106.2363776, 106.5044223, 106.5048439, 106.8777876, 106.8781188, 107.0560524, 108.7036241, 108.4896401, 108.7160024, 108.5445139, 108.6752535, 109.016092, 109.2828719, 110.2925573, 109.6029595, 109.8184794, 109.8293217, 111.4162672, 113.9612468, 111.2189682, 111.0556717, 111.3428109, 111.6008567, 117.632707, 112.3263545, 112.3776443, 112.5576259, 113.0883244, 113.6265221, 113.6913046, 114.138578, 114.4322225, 115.2125941, 115.3263174, 116.3572958, 115.2506023, 115.2825568, 115.5762013, 116.3388338, 116.2405307, 116.2717926, 116.9716303, 116.6803048, 116.6086555, 117.2900314, 117.889277, 117.5611783, 118.0644698, 117.9584567, 118.0885939, 123.0520299, 119.5683816, 119.1119225, 119.4074645, 119.6395792, 119.9171411, 120.1221501, 120.7169384, 120.7177817, 120.9979938, 121.3430187, 121.5671523, 122.3437593, 121.6905734, 122.6455656, 123.2027975, 123.1537664, 123.5702297, 123.4759321, 123.4791246, 124.0378623, 124.6159656, 124.4878162, 125.1394661, 125.0590526, 125.6645929, 125.3605278, 125.9360108, 126.3346144, 126.4411095, 126.1866176, 127.1610355, 126.6626531, 127.3787541, 127.3291509, 127.8193716, 127.9235175, 128.1348813, 128.4792737, 128.8527293, 129.2337444, 129.5052225, 130.9839863, 129.721887, 130.2989964, 130.8217138, 130.3274574, 130.3743501, 130.5914964, 131.593502, 131.6687954, 132.1307661, 132.3414673, 132.8912603, 132.8268392, 133.9066681, 133.5985069, 133.7197294, 134.2875325, 134.7458289, 135.7971365, 135.0901609, 135.2027699, 135.5290014, 135.6707639, 135.9218224, 135.8866755, 136.5495291, 136.9980674, 136.7636336, 136.9271108, 137.2173822, 142.0290567, 137.4830477, 138.2129727, 142.3017395, 139.1124285, 139.0858951, 138.9539508, 139.6953807, 139.5247356, 139.7924189, 139.7664878, 140.4653317, 140.5355655, 140.951065, 140.9757913, 141.6594561, 141.5882586, 142.1251612, 141.8095009, 142.4965087, 142.3700457, 146.2194395, 146.0136775, 146.4010775, 147.4901825, 150.7779773, 170.6315706, 195.7956125, 151.9306903, 153.1871576, 156.3636084, 173.1025518, 149.2943647, 150.7201519, 154.1707312, 147.1223889, 147.314327, 148.9001281, 148.3246149, 148.6790664, 148.606604, 149.1337486, 148.8417907, 149.551778, 149.52666, 150.2175529, 150.0986496, 151.2831965, 150.9871125, 195.0743614, 162.7430134, 157.8442395]
    # Start timing 2
    # [34154.9201121]


def cprofileTest():
    dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    database = testParametersNP.getDatabase(dataframe)
    mesh_ = vedo.load('DB_fixed/Human/13.ply')
    profiler = cProfile.Profile()
    profiler.enable()
    #
    # for t in range(0, 10):
    #     for i in range(0, len(database)):
    #         mostSimilar = qHF.getQueryResults(database, i)
    step4.calculateFeatures2(mesh_)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

if __name__ == '__main__':
    # main()
    # computeTime()
    cprofileTest()
    # computeAndPlotROCModels()
    # dataframe = step4.readCSVAsDataFrame('featuresNew.csv')
    # compareDistMetrics([5, 10, 20, 40], dataframe)
    # compAndPlotMAP(minQuerySize=1, maxQuerySize=6)
    # plotMetric(10, 0)
    # plotMetric(20, 0)
    # plotMetric(10, 1)
    # plotMetric(40, 1)
    # plotMetric(10, 2)
    # plotMetric(20, 2)
    # sV.weightsGlobalFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
    # sV.weightsShapeFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
    # sV.weightsGlobalAndShape = [0.5, 0.5]
    # computeAndPlotROC(True, 0)
    # testje()
    # makeROCTable()