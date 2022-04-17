import itertools
import math
import vedo
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import barycenter as bc
from time import perf_counter


# return a random number, not present in 'exclude' already, between start and stop
def getRandom(stop: int, exclude: List[int], start: int = 0):
    ans = random.randint(start, stop)
    if ans in exclude:
        return getRandom(stop, exclude, start)
    else:
        return ans


# Give control point (pc), points 1 and 2: compute the angle between them in radians
def calcAngle(pc: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    c1 = p1 - pc
    c2 = p2 - pc

    cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
    angle = np.arccos(cos)
    if cos < -1 or cos > 1:
        print(f"Cos is now: {cos}")
        angle = np.arccos(-1) if cos < -1 else np.arccos(1)
    return angle


# Compute A3: Given a mesh, and a number of samples: compute the angle between 3 random vertices
def A3(mesh_: vedo.mesh.Mesh, samples: int = 1000):
    ans = []
    maxx = math.ceil(samples ** (1/3))
    vertices = mesh_.vertices()
    aantal = len(vertices)-1
    for i in range(0, maxx):
        v1i = random.randint(0, aantal)
        for j in range(0, maxx):
            v2i = getRandom(aantal, [v1i])
            for k in range(0, maxx):
                v3i = getRandom(aantal, [v1i, v2i])
                # Do the real calculating
                angle = calcAngle(vertices[v1i], vertices[v2i], vertices[v3i])
                ans.append(angle)
    return ans


# Compute D1: given a mesh and number of samples, compute the distance between random vertices and the origin
def D1(mesh_: vedo.mesh.Mesh, samples: int = 1000):
    """squared = np.sum((p1-p2)**2, axis=0)
    return math.sqrt(squared)"""
    ans = []
    vertices = mesh_.vertices()
    aantal = len(vertices)-1
    for i in range(0, samples):
        v1i = random.randint(0, aantal)
        ans.append( math.sqrt(np.sum((vertices[v1i])**2, axis=0)) )
    return ans


# compute D2: given a mesh and number of samples, compute the distance between 2 randomly selected vertices
def D2(mesh_: vedo.mesh.Mesh, samples: int = 1000):
    ans = []
    maxx = math.ceil(samples ** (1/2))
    vertices = mesh_.vertices()
    aantal = len(vertices)-1
    for i in range(0, maxx):
        v1i = random.randint(0, aantal)
        for j in range(0, maxx):
            v2i = getRandom(aantal, [v1i])
            ans.append(bc.calcLength(vertices[v1i], vertices[v2i]))
    return ans


# compute D3: given mesh and number of samples, compute the area of triangles consisting of randomly selected vertices
def D3(mesh_: vedo.mesh.Mesh, samples: int = 1000):
    ans = []
    maxx = math.ceil(samples ** (1/3))
    vertices = mesh_.vertices()
    aantal = len(vertices)-1
    for i in range(0, maxx):
        v1i = random.randint(0, aantal)
        for j in range(0, maxx):
            v2i = getRandom(aantal, [v1i])
            for k in range(0, maxx):
                v3i = getRandom(aantal, [v1i, v2i])
                # Do the real calculating
                punten = bc.getDistancesDecimal([vertices[v1i], vertices[v2i], vertices[v3i]])
                l = bc.calcTriangleAreaDecimal(punten[0], punten[1], punten[2])
                ans.append(float(l.sqrt()))
    return ans


# Compute d4: given a mesh and number of samples: compute the volume of tetrahedrons consistint of random vertices
def D4(mesh_: vedo.mesh.Mesh, samples: int = 1000):
    ans = []
    maxx = math.ceil(samples ** (1/4))
    vertices = mesh_.vertices()
    aantal = len(vertices)-1
    one_sixth = 1/6
    one_third = 1/3
    for i in range(0, maxx):
        v1i = random.randint(0, aantal)
        for j in range(0, maxx):
            v2i = getRandom(aantal, [v1i])
            for k in range(0, maxx):
                v3i = getRandom(aantal, [v1i, v2i])
                for l in range(0, maxx):
                    v4i = getRandom(aantal, [v1i, v2i, v3i])

                    # Do the real calculating
                    v2, v3, v4 = vertices[v2i] - vertices[v1i], vertices[v3i] - vertices[v1i], vertices[v4i] - vertices[v1i]
                    cr = np.cross(v2, v3)
                    d = np.dot(cr, v4)
                    ans.append( ((abs(d) * one_sixth)**one_third) )
    return ans


# Compute the compactness for a mesh, given the surface area of that mesh
def compCompactness(mesh_: vedo.mesh.Mesh, surfaceArea: float):
    volume2 = calcVolume(mesh_)**2
    c = (surfaceArea**3)/(36*math.pi*volume2)
    return c


# Given the vertices of model, a face, index: return the vertex
def getVertex(vertices, face, index, forNeg1):
    return forNeg1 if index == -1 else vertices[face[index]]


# Compute the volume of a mesh: compute the volume of tetrahedrons consisting of the barycenter and a face (For each face)
def calcVolume(mesh_: vedo.mesh.Mesh):
    faces = mesh_.faces()
    total = 0
    one_sixth = 1/6
    vertices = mesh_.vertices()
    opties = [-1, 0, 1, 2]
    # opties = [0, 1, 2, -1]
    orig = np.array([0.0, 0.0, 0.0])
    for f in faces:
        # print(f)
        a = 0
        for s in itertools.combinations(opties, 3):
            cr = np.cross(getVertex(vertices, f, s[0], orig), getVertex(vertices, f, s[1], orig))
            d = np.dot(cr, getVertex(vertices, f, s[2], orig))
            a += d
        total += a
        # print(f"Face: {f} with {a}")
    v = abs(total) * one_sixth
    # print(f"Volume: {v}")
    return v


# given a mesh and the name of the shape feature, return the computed distribution for that feature
def getFeatureResult(mesh_: vedo.mesh.Mesh, feature: str):
    if feature == "A3":
        return A3(mesh_, 100000)
    elif feature == "D1":
        return D1(mesh_, 1000)
    elif feature == "D2":
        return D2(mesh_, 100000)
    elif feature == "D3":
        return D3(mesh_, 40000)
    elif feature == "D4":
        return D4(mesh_, 40000)


# Given a feature, and number of bins, give the bins based on max possible value
def getBins(feature: str, bins: int):
    epsilon = 0.000000001
    if feature == "A3":
        return np.arange(0.0, math.pi+epsilon, (math.pi/bins))
    elif feature == "D1":
        return np.arange(0.0, math.sqrt(3)+epsilon, (math.sqrt(3)/bins))
    elif feature == "D2":
        return np.arange(0.0, math.sqrt(3)+epsilon, (math.sqrt(3)/bins))
    elif feature == "D3":
        return np.arange(0.0, 0.93061+epsilon, (0.93061/bins))
    elif feature == "D4":
        return np.arange(0.0, 0.5504+epsilon, (0.5504/bins))


# given the number of bins and the maximum value (ACTUALLY PRESENT VALUE), return the bins for making the histogram
def getBinsN(bins: int, maxW: int):
    epsilon = 0.000000001
    return np.arange(0.0, maxW+epsilon, (maxW/bins))


# given the database location, the save location for the figures, the bincounts to use:
# Compute all features and make the histograms
def getAllFeatures2(database: str = 'DB_scale', saveLoc: str = 'figures2', binCount: list = [25, 25, 25, 25, 25]):
    subfolders = [f.path for f in os.scandir(database) if f.is_dir()]
    features = [[], [], [], [], []]
    featureLabels = ["A3", "D1", "D2", "D3", "D4"]
    # bins = [getBins(featureLabels[i], binCount[i]) for i in range(0, 5)]
    # print(bins)
    # subfolders = ["DB_scale\\Airplane"]
    # featureLabels = ["A3"]
    maxima = [0.0, 0.0, 0.0, 0.0, 0.0]
    for fl_id, fl in enumerate(featureLabels):
        feature = []
        print(f"Feature {fl}")
        for subfolder in subfolders:
            klasse = []
            print(f"Subfolder: {subfolder}")
            for file in os.listdir(subfolder):
                if file.endswith(".off") or file.endswith(".ply"):
                    # print(f"File {file}")
                    mesh_ = vedo.load(subfolder + '/' + file)

                    ans = getFeatureResult(mesh_, fl)
                    klasse.append(ans)
                    maxAns = max(ans)
                    if maxAns > maxima[fl_id]:
                        maxima[fl_id] = maxAns
            feature.append(klasse)
        features[fl_id] = feature
    sbfs = [x.split('\\')[1] for x in subfolders]

    features2 = []
    maxMinAvg =[]
    for f_id, f in enumerate(features):
        f_ = []
        mma = []
        for k in f:
            k_ = []
            k_max, k_min, k_avg = [], [], []
            for i in k:
                density, bins, _ = plt.hist(i, bins=getBinsN(binCount[f_id], maxima[f_id]), weights=np.ones(len(i)) / len(i))
                X = [(bins[i] + bins[i + 1]) / 2 for i in range(0, binCount[f_id])]
                k_.append((X, density))
                k_max.append(max(i))
                k_min.append(min(i))
                k_avg.append(sum(i)/len(i))
            f_.append(k_)
            mma.append((max(k_max), min(k_min), sum(k_avg)/len(k_avg)))
        features2.append(f_)
        maxMinAvg.append(mma)

    # [A3:[Ant:[ant_i:(X, Y), ...], ....], ...]
    ylims = [max([max([max(ind[1]) for ind in k]) for k in f])*1.1 for f in features2]
    plt.close()
    for f_id, feature in enumerate(features2):
        for c_id, klasse in enumerate(feature):
            for indiv in klasse:
                print(f"Lengte X: {len(indiv[0])}, Y: {len(indiv[1])}")
                print(indiv[1])
                plt.plot(indiv[0], indiv[1])
            plt.ylim(0.0, ylims[f_id])
            plt.title(f"{featureLabels[f_id]} for class: {sbfs[c_id]} - min: {'{:.4f}'.format(maxMinAvg[f_id][c_id][1])},"
                      + f" max: {'{:.4f}'.format(maxMinAvg[f_id][c_id][0])}, \u03BC: "
                      + f"{'{:.4f}'.format(maxMinAvg[f_id][c_id][2])}")
            plt.tight_layout()
            plt.savefig(f'{saveLoc}/{featureLabels[f_id]}_{sbfs[c_id]}.png')
            plt.close()
    print(f"The maxima of the features are: {maxima}")
    print("Done")



#
#
# Below is all for testing ONLY
#
#

#
def getAllFeatures(database: str = 'DB_scale', saveLoc: str = 'figures2', binCount: list = [25, 25, 25, 25, 25]):
    subfolders = [f.path for f in os.scandir(database) if f.is_dir()]
    features = [[], [], [], [], []]
    featureLabels = ["A3", "D1", "D2", "D3", "D4"]
    # bins = [getBins(featureLabels[i], binCount[i]) for i in range(0, 5)]
    # print(bins)
    # subfolders = ["DB_scale\\Airplane"]
    for fl_id, fl in enumerate(featureLabels):
        feature = []
        print(f"Feature {fl}")
        for subfolder in subfolders:
            klasse = []
            print(f"Subfolder: {subfolder}")
            for file in os.listdir(subfolder):
                if file.endswith(".off") or file.endswith(".ply"):
                    # print(f"File {file}")
                    mesh_ = vedo.load(subfolder + '/' + file)

                    ans = getFeatureResult(mesh_, fl)

                    density, bins, _ = plt.hist(ans, bins=getBins(fl, binCount[fl_id]), weights=np.ones(len(ans)) / len(ans))
                    X = [(bins[i]+bins[i+1])/2 for i in range(0, binCount[fl_id])]
                    klasse.append((X, density))
            feature.append(klasse)
        features[fl_id] = feature
    sbfs = [x.split('\\')[1] for x in subfolders]
    # [A3:[Ant:[ant_i:(X, Y), ...], ....], ...]
    ylims = [max([max([max(ind[1]) for ind in k]) for k in f])*1.1 for f in features]
    plt.close()
    for f_id, feature in enumerate(features):
        for c_id, klasse in enumerate(feature):
            for indiv in klasse:
                print(f"Lengte X: {len(indiv[0])}, Y: {len(indiv[1])}")
                print(indiv[1])
                plt.plot(indiv[0], indiv[1])
            plt.ylim(0.0, ylims[f_id])
            plt.title(f"{featureLabels[f_id]} for class: {sbfs[c_id]}")
            plt.tight_layout()
            plt.savefig(f'{saveLoc}/{featureLabels[f_id]}_{sbfs[c_id]}.png')
            plt.close()
    print("Done")



def extractFeatureTest():
    mesh_ = vedo.load('DB/Human/5.off')
    print("Start A3")
    t1 = perf_counter()
    A3(mesh_, 100000)
    t2 = perf_counter()
    print(f"Time {t2-t1}")

    print("Start D1")
    t1 = perf_counter()
    D1(mesh_, 1000)
    t2 = perf_counter()
    print(f"Time {t2-t1}")

    print("Start D2")
    t1 = perf_counter()
    D2(mesh_, 100000)
    t2 = perf_counter()
    print(f"Time {t2-t1}")

    print("Start D3")
    t1 = perf_counter()
    D3(mesh_, 40000)
    t2 = perf_counter()
    print(f"Time {t2-t1}")

    print("Start D4")
    t1 = perf_counter()
    D3(mesh_, 40000)
    t2 = perf_counter()
    print(f"Time {t2-t1}")




def testMain():
    # mesh_ = vedo.load('DB/cube3triangles.off')
    # mesh_ = vedo.load('DB/cube3trianglesMix.off')
    # mesh_ = vedo.load('DB/tetrahedron.off')
    mesh_ = vedo.load('testDB/donut3_3counterClock.off')

    # testX()
    # normals = mesh_.normals(cells=True, compute=True)
    # print(normals)
    # mesh_.flipNormals()
    # print(mesh_.normals(cells=True, compute=True))
    calcVolume(mesh_)
    mesh_.frontFaceCulling(True)
    # print(mesh_.faces())
    # print(len(mesh_.faces()))

    mesh_.show(axes=8)
    # testje()


def testX():
    # punten = [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
    punten = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    punten = [np.array(x) for x in punten]
    som = 0
    for s in itertools.combinations(punten, 3):
        # print(s)
        cr = np.cross(s[0], s[1])
        d = np.dot(cr, s[2])
        print(f"cross: {cr} ==> {d}")
        som += d
    print(f"Volume: {abs(som)/6}")


def calcArea(a, b, c):
    x = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    if x > 0.0:
        return 0.25 * math.sqrt(x)
    else:
        return 0.0


def testje():
    ans = 0.0
    p1 = np.array([0.0, 0.0, 0.0])
    for i in np.arange(0.0, 1.00000001, 0.1):
        for j in np.arange(0.0, 1.00000001, 0.1):
            for k in np.arange(0.0, 1.00000001, 0.1):
                for a in np.arange(0.0, 1.00000001, 0.1):
                    for b in np.arange(0.0, 1.00000001, 0.1):
                        for c in np.arange(0.0, 1.00000001, 0.1):
                            p2 = np.array([i, j, k])
                            p3 = np.array([a, b, c])
                            punten = bc.getDistances([p1, p2, p3])
                            if punten[0] <= 0.0 or punten[1] <= 0.0 or punten[2] <= 0.0:
                                continue
                            else:
                                l = calcArea(punten[0], punten[1], punten[2])
                                if l > ans:
                                    ans = l
                                    print(f"Biggest: {ans} with ({i}, {j}, {k}) & ({a}, {b}, {c})")

def testenDingen():
    # x = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.2]
    # d, b, _ = plt.hist(x, bins=[0.0, 0.11, 0.22, 0.33, 0.44, 0.55], weights=np.ones(len(x))/len(x))
    # plt.show()
    subfolders = [f.path for f in os.scandir('DB') if f.is_dir()]
    sbfs = [x.split('\\')[1] for x in subfolders]
    print(sbfs)

    mesh_ = vedo.load('DB/Human/5.off')
    print("Start a3")
    a3 = A3(mesh_, 100000)
    # normalisation.makeHistogram2(a3, "A3 human5", "Angle 3 vertices")
    density, bins, _ = plt.hist(a3, bins=getBins("A3", 25), weights=np.ones(len(a3))/len(a3))
    print(density)
    print(bins)
    print(len(bins))
    print(len(density))
    X = [(bins[i]+bins[i+1])/2 for i in range(0, 25)]
    plt.close()
    plt.plot(X, density)
    plt.tight_layout()
    plt.show()


# print(np.arccos(-1.000001))
# print(np.arccos(-1))
# getAllFeatures2()
# extractFeatureTest()
# testenDingen()


# NOT USED ANYMORE
# def calcAngle2(pc: list, p1: list, p2: list):
#     h1 = (p1[0] - pc[0]) * (p2[0] - pc[0]) + (p1[1] - pc[1]) * (p2[1] - pc[1]) + (p1[2] - pc[2]) * (p2[2] - pc[2])
#     h2 = math.sqrt( (p1[0] - pc[0])**2 + (p1[1] - pc[1])**2 + (p1[2] - pc[2])**2) * \
#          math.sqrt((p2[0] - pc[0])**2 + (p2[1] - pc[1])**2 + (p2[2] - pc[2])**2)
#     angle = math.acos(h1/h2)
#     # print(angle)
#     # print(angle * 180 / math.pi)
#     return angle