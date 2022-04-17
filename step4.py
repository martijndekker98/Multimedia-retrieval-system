import vedo
import csv
import numpy as np
import pandas
from typing import List
import matplotlib.pyplot as plt
from barycenter import findBaryCenter2
from pcaFunctions import pca
from normalisation import subAndSuperSample, translationNormalisation, scale
from three_features import flipping_test, eccentricity, get_surface_area, get_bounding_box_volume, diameter
from meshFixing import meshFixingFaceOrientationAndHoles
from featureExtraction import A3, D1, D2, D3, D4, compCompactness, getBins, getBinsN
import staticVariables as sV


# given a mesh, preprocess it
def preprocessMesh(mesh_: vedo.mesh.Mesh):
    # Normalise number of vertices
    subAndSuperSample(mesh_, sV.numbVerticesGoal, margin=2)

    # Normalise position
    x, y, z = findBaryCenter2(mesh_)
    translationNormalisation(mesh_, [x, y, z])

    # PCA
    new_vertices = pca(mesh_, True)
    new_mesh = mesh_.points(new_vertices)

    # Flipping test
    new_mesh = flipping_test(new_mesh)

    # Normalise scale
    scale(new_mesh)

    return new_mesh


# Fix the given mesh before computing features (stitch holes, fix oriertations and protrusions)
def fixMeshForFeatures(mesh_: vedo.mesh.Mesh):
    return meshFixingFaceOrientationAndHoles(mesh_)


# Write a mesh to locationAndName
def saveMesh(mesh_: vedo.mesh.Mesh, locationAndName: str):
    vedo.write(mesh_, locationAndName)


# calculate the features of a mesh
def calculateFeatures2(mesh_: vedo.mesh.Mesh):
    # print("Did you make sure that the mesh has been preprocessed AND fixed holes/normals? Process will continue now")
    surfaceArea = get_surface_area(mesh_)

    compactness = compCompactness(mesh_, surfaceArea)
    boundingBoxVol = get_bounding_box_volume(mesh_)

    diame = float(diameter(mesh_))
    eccen = eccentricity(mesh_)

    globalDescriptors = [surfaceArea, compactness, boundingBoxVol, diame, eccen]

    # Shape property descriptors
    a3 = A3(mesh_, sV.A3samples)
    d1 = D1(mesh_, sV.D1samples)
    d2 = D2(mesh_, sV.D2samples)
    d3 = D3(mesh_, sV.D3samples)
    d4 = D4(mesh_, sV.D4samples)
    shapef = [a3, d1, d2, d3, d4]

    # shapeFeatures = []
    # for f_id, feat in enumerate(shapef):
    #     bins1 = getBins(sV.featureLabels[f_id], sV.binCount[f_id])
    #     density, bins, _ = plt.hist(feat, bins=bins1, weights=np.ones(len(feat)) / len(feat))
    #     # density = y waarde, oftewel percentage van samples (van feature) in een bin.
    #     shapeFeatures.append(density)
    globalDescriptors.extend(shapef)
    return globalDescriptors


# Get the maximum values per shape feature
def getMaxima(modelsFeatures: list):
    maxima = [0.0, 0.0, 0.0, 0.0, 0.0]
    averages = []
    for m in modelsFeatures:
        # avg_ = []
        # for i in range(0, 5):
        #     if i == 0: print(f"m[i]: {type(m[i])}")
        #     avg_.append(sum(m[i])/len(m[i]))
        for i in range(5, len(m)):
            max_ = max(m[i])
            if max_ > maxima[i-5]:
                maxima[i-5] = max_
        # averages.append(avg_)
    return maxima, averages


# Compute the histograms from the features
def computeHistogramsFromFeatures(modelsFeatures: list):
    ans = []
    maxima, averages = getMaxima(modelsFeatures)
    print(f"The maxima are: {maxima}")
    for model in modelsFeatures:
        nieuwe = model[:5]
        for i in range(5, len(model)):
            bins1 = getBinsN(sV.binCount[i-5], maxima[i-5])
            density, bins, _ = plt.hist(model[i], bins=bins1, weights=np.ones(len(model[i])) / len(model[i]))
            # density = y waarde, oftewel percentage van samples (van feature) in een bin.
            nieuwe.extend(density)
        ans.append(nieuwe)
    return ans, maxima #, averages


# write the extracted features to a csv file
def writeToCsv(file: str, header: List, data: List):
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        if len(header) > 0:
            writer.writerow(header)

        writer.writerows(data)


# read a csv file and turn it into a pandas dataframe
def readCSVAsDataFrame(file: str):
    df = pandas.read_csv(file)
    return df


#
#
# Below for testing only
#
#

"""Deze moet nog gecontroleerd worden"""
def calculateFeatures(mesh_: vedo.mesh.Mesh):
    print("Did you make sure that the mesh has been preprocessed AND fixed holes/normals?")
    surfaceArea = get_surface_area(mesh_)

    # Werkt nog niet!
    compactness = compCompactness(mesh_, surfaceArea)
    boundingBoxVol = get_bounding_box_volume(mesh_)

    # Kunnen niet dezelfde naam hebben als methode naam
    diame = diameter(mesh_)
    eccen = eccentricity(mesh_)

    globalDescriptors = [surfaceArea, compactness, boundingBoxVol, diame, eccen]


    # Shape property descriptors
    a3 = A3(mesh_, sV.A3samples)
    d1 = D1(mesh_, sV.D1samples)
    d2 = D2(mesh_, sV.D2samples)
    d3 = D3(mesh_, sV.D3samples)
    d4 = D4(mesh_, sV.D4samples)
    shapef = [a3, d1, d2, d3, d4]
    # Moeten nog in histograms gepropt worden... maar de bins zijn afhankelijk van alle models in de database?
    # IDK wat we daar mee moeten.
    # TIJDELIJKE OPLOSSING:
    shapeFeatures = []
    for f_id, feat in enumerate(shapef):
        bins1 = getBins(sV.featureLabels[f_id], sV.binCount[f_id])
        density, bins, _ = plt.hist(feat, bins=bins1, weights=np.ones(len(feat)) / len(feat))
        # density = y waarde, oftewel percentage van samples (van feature) in een bin.
        shapeFeatures.append(density)


    for i in shapeFeatures:
        globalDescriptors.extend(i)
    return globalDescriptors

#
# mesh_ = vedo.load('DB_fixed/Human/14.ply')
# print("Loaded")
# t1 = time.perf_counter_ns()
# features = calculateFeatures2(mesh_)
# t2 = time.perf_counter_ns()
# print(f"Time: {(t2-t1)/1000000} ms")