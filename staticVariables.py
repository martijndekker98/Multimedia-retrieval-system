#
# Contains some variables used throughout the system
#

# Just some variables
A3samples = 100000
D1samples = 1000
D2samples = 100000
D3samples = 40000
D4samples = 40000

numbVerticesGoal = 1500

featureLabels = ["A3", "D1", "D2", "D3", "D4"]
binCountOld = [25, 25, 25, 25, 25]
binCount = [25, 10, 25, 25, 25]

# Related to the CSV file
featuresCSV = 'DB_fixed/features.csv'
startGlobal = 2
endGlobal = startGlobal+5
className = 0
fileName = 1

# Related to metric calculations
modelsTotal = 380
maxModelsQuery = modelsTotal - 1
histDistPickle = 'distancesHists.pickle'

# Related to the distance computation
weightsGlobalFeatures = [0.47, 0.39, 0.06, 0.01, 0.08]
weightsShapeFeatures = [0.16, 0.21, 0.31, 0.21, 0.11]  # [1.0, 1.0, 1.0, 1.0, 1.0]
weightsGlobalAndShape = [0.65, 0.35]
# weightsGlobalFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]
# weightsShapeFeatures = [1.0, 1.0, 1.0, 1.0, 1.0]  # [1.0, 1.0, 1.0, 1.0, 1.0]
# weightsGlobalAndShape = [0.5, 0.5]
useEuclidean = True
useAngular = True
useWeightedVersion = False      # For angular/cosine

# Normalisation histograms
histDistsFileStart = 'testDB/hist_dist_'
histDistAvgs = [0.0, 0.0, 0.0, 0.0, 0.0]
histDistStds = [0.0, 0.0, 0.0, 0.0, 0.0]
calculateAvgAndStd = False

def getHeader():
    # ans = ["Surface_Area", "Compactness", "Bound_Box_vol", "Diameter", "Eccentricity"]
    ans = ["Subfolder", "File_name", "Surface_Area", "Compactness", "Bound_Box_vol", "Diameter", "Eccentricity"]
    for f_id, f in enumerate(featureLabels):
        for b in range(0, binCount[f_id]):
            ans.append(f"{f}_bin{b}")
    return ans


import pandas
def getAvgAndStd(dataframe: pandas.DataFrame):
    # print(dataframe.shape)
    assert dataframe.shape[0] == dataframe.shape[1]
    import numpy as np
    values = []
    for i in range(0, dataframe.shape[0]):
        for j in range(i+1, dataframe.shape[1]):
            values.append(dataframe.iloc[i, j])
    nparray = np.array(values)
    return np.average(nparray), np.std(nparray)


def extractDistsFromFile():
    for f_id, feat in enumerate(featureLabels):
        csvFile = pandas.read_csv(f"{histDistsFileStart}{feat}.csv", header=None)
        # print(csvFile)
        avg, std = getAvgAndStd(csvFile)
        histDistAvgs[f_id] = avg
        histDistStds[f_id] = std

if calculateAvgAndStd:
    extractDistsFromFile()
    print(histDistAvgs)
    print(histDistStds)
else:
    histDistAvgs = [0.0068932322691584785, 0.036836247743368966, 0.00834368012630391, 0.011425554445584747, 0.014814713956402943]
    histDistStds = [0.003830834506112367, 0.022344829190635504, 0.004344245566992107, 0.007520522449538178, 0.00949786976083969]
