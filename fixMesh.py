import vedo
import os
import numpy as np
from queue import SimpleQueue
import barycenter


# Given a mesh, construct the dictionary to look up. Key: edge (as tuple), the value will be the face IDs that contain that edge
def makeMeshLookup(mesh_: vedo.mesh.Mesh):
    faces = mesh_.faces()
    dict_ = {} # key = edge as tuple, value = face id
    opties = [0, 1, 2, 0]
    dubbele = 0
    problematic = set()
    for id, face in enumerate(faces):
        for i in range(0, len(face)):
            try:
                dict_[(face[opties[i]], face[opties[i+1]])].append(id)
                print(f"ADDED! {face[opties[i]]} <-> {face[opties[i+1]]}")
                dubbele += 1
                problematic.add((face[opties[i]], face[opties[i+1]]))
            except KeyError as err:
                dict_[(face[opties[i]], face[opties[i+1]])] = [id]
    # print(dict_.values())
    # print(dict_.keys())
    print(f"Number of faces: {len(faces)}, vertices: {len(mesh_.vertices())}, edges: {len(dict_.keys())}")
    return dict_, dubbele, problematic


# Given a list of faces, find the neighbour of a face (current)
def findNeighbour(lijst: list, current: int, startFrom: int):
    print(f"LIJST: {lijst}")
    if len(lijst) > 2:
        print(f"ERROR, LIST IS TOO LONG: {lijst}")
    for i in range(0, len(lijst)):
        if lijst[i] != current and lijst[i] >= startFrom:
            return lijst[i]
    return -1


# Add the new faces to the dictionary, starting from an index (index corresponding to the list of faces).
# Before hole stitching: 10 faces, afterward: 12; start from the 10th face, and add all the faces afterward to the dict
def addNewFacesToDict(dict_: dict, faces: list, startFrom: int):
    opties = [0, 1, 2, 0]
    for i in range(startFrom, len(faces)):
        f = faces[i]
        print(f"face {i} with {f}")
        for j in range(0, 3):
            try:
                dict_[(f[opties[j]], f[opties[j+1]])].append(i)
                # print(f"Added, {(f[opties[j]], f[opties[j+1]])} == {i}")
            except KeyError as err:
                dict_[(f[opties[j]], f[opties[j+1]])] = [i]
                # print(f"New list, {(f[opties[j]], f[opties[j+1]])} == {i}")
    return dict_


# Find the faces in the mesh (using the dictionary) that are oriented wrong. Then fix the orientation of those faces
def findWrongFaces2(faces: list, dict_: dict, startFrom: int = 0, startFace: int = 0):
    isCorrect = [False] * len(faces)
    opties = [0, 1, 2, 0]

    toDo = SimpleQueue()
    toDo.put_nowait(startFace)
    # isCorrect[0] = True
    for i in range(0, 1):
        isCorrect[i] = True
    # if startFrom > 0:
    #     dict_ = addNewFacesToDict(dict_, faces, startFrom)
    fixed = []
    while not toDo.empty():
        current = toDo.get_nowait()
        # Check neighbours using the edges + dict_
        # print(f"Current: {current} >> {len(faces[current])}")
        for i in range(0, len(faces[current])):
            keyNeigh = (faces[current][opties[i+1]], faces[current][opties[i]])
            try:
                # print(f"Looking up: {keyNeigh} {dict_[keyNeigh]}")
                neighbourID = dict_[keyNeigh][0]
            except KeyError as err:  # fix neighbour
                neighbourID = findNeighbour(dict_[(faces[current][opties[i]], faces[current][opties[i+1]])], current, startFrom)
                # print(f"KeyNeigh: {keyNeigh}, Current: {current} and n ID: {neighbourID} >> {faces[current]} <> {faces[neighbourID]};  dict: {dict_[(faces[current][opties[i]], faces[current][opties[i+1]])]}")
                if neighbourID != -1 and not isCorrect[neighbourID]:
                    faceBef = faces[neighbourID]
                    faces[neighbourID] = [faces[neighbourID][0], faces[neighbourID][2], faces[neighbourID][1]]
                    print(f"Fixed face {neighbourID} from [{faces[neighbourID][0]}, {faces[neighbourID][2]}, {faces[neighbourID][1]}] to {faces[neighbourID]}")
                    updateDict(dict_, faceBef, opties, neighbourID, faces[neighbourID])
                    fixed.append(neighbourID)
                elif neighbourID == -1:
                    print(f"NeighbourID is -1: {dict_[(faces[current][opties[i]], faces[current][opties[i+1]])]}, current: {current}")

            # print(f"Neighbour ID: {neighbourID}")
            if neighbourID != -1 and not isCorrect[neighbourID]:
                toDo.put_nowait(neighbourID)
                isCorrect[neighbourID] = True
    fixed.sort()
    print(f"Fixed faces: {fixed}")
    return faces


# Update the dictionary (after orientation was fixed)
def updateDict(dict_: dict, faceBef: list, opties: list, neighbourID: int, faceN: list):
    # Fix faces in dict
    for i in range(0, 3):
        if len(dict_[(faceBef[opties[i]], faceBef[opties[i + 1]])]) < 2:
            del dict_[(faceBef[opties[i]], faceBef[opties[i + 1]])]
        else:
            dict_[(faceBef[opties[i]], faceBef[opties[i + 1]])].remove(neighbourID)
        try:
            dict_[(faceN[opties[i]], faceN[opties[i + 1]])].append(neighbourID)
        except KeyError as err:
            dict_[(faceN[opties[i]], faceN[opties[i + 1]])] = [neighbourID]
        print(
            f"Remove edge ({faceBef[opties[i]]}, {faceBef[opties[i + 1]]}) for ({faceN[opties[i]]}, {faceN[opties[i + 1]]})")
    # print(faces[neighbourID])


# Find the missing edges in the dictionary. If an edge A B is only present as A B in the dictionary, B A is missing
def findMissingEdges(dict_: dict):
    keys = dict_.keys()
    missing = []
    for key in keys:
        if len(dict_[key]) == 1:
            try:
                dict_[(key[1], key[0])]
            except KeyError as err:
                missing.append((key[1], key[0]))
    return missing


# Find the 'False' element in the list
def findFalse(lijst: list):
    for i in range(0, len(lijst)):
        if not lijst[i]:
            return i
    return -1


# Find an element in a list
def findInMissing(missing: list, key: int):
    for id, pair in enumerate(missing):
        if pair[0] == key:
            return id
    print(f"ERROR IN findInMissing: It seems like there is no pair with key as its first value; key: {key}")
    print(missing)
    return -1


# Find loops in the missing edges, to construct loops of the holes present in the mesh
def findLoopsInEdges(missing: list):
    loops = []
    processedC = 0
    processed = [False] * len(missing)
    while True:
        if processedC == len(missing):
            break
        first = findFalse(processed)
        if first == -1:
            print(f"Error, -1 ")
            break
        loop = [missing[first]]
        processed[first] = True
        key = loop[-1][1]
        processedC += 1
        print(f"Key: {key}, loop[0][0]: {loop[0][0]} <<>> {loop}")
        while key != loop[0][0]:
            # find point and add to the list
            index = findInMissing(missing, key)
            processed[index] = True
            processedC += 1
            loop.append(missing[index])
            key = missing[index][1]
        loops.append(loop)
        print(f"Loop: {loop}")
    return loops


# Delete the false points in the mesh
def deleteFalsePoints(mesh_: vedo.mesh.Mesh, dict_: dict, prob: set):
    facesToCheck = []
    print(f"Prob deleteFalsePoints before: {prob}")
    for i in prob:
        facesToCheck.extend(dict_[(i[0], i[1])])
        try:
            facesToCheck.extend(dict_[(i[1], i[0])])
        except KeyError as err: pass
    verticesToCheck = set([item for i in facesToCheck for item in mesh_.faces()[i]])
    verticesChecked = []
    for vertex in verticesToCheck:
        aantalKeer = 0
        facesPresent = []
        for f in mesh_.faces():
            if vertex in f:
                aantalKeer += 1
                facesPresent.append(f)
        if aantalKeer < 3:
            verticesChecked.append((vertex, facesPresent))
        print(f"Vertex {vertex} appears in {aantalKeer} faces")
    facesToRemove = []  # IDs
    facesToAdd = []  # faces: [x, y, z]
    for vc in verticesChecked:
        print(f"Vertex {vc[0]} is present in {vc[1]}")
        # lijnen = [a for x in vc[1] for a in x if a != vc[0]]
        lijnen2 = [[a for a in x if a != vc[0]] for x in vc[1]]
        # print(f"To remove edge: {lijnen}")
        print(f"To remove edge: {lijnen2}")
        for lijn in lijnen2:
            print(f"The line: {lijn}")
            inFaces = []
            for optie in facesToCheck:
                # print(f"Line: {lijn} <> option: {mesh_.faces()[optie]}")
                if lijn[0] in mesh_.faces()[optie] and lijn[1] in mesh_.faces()[optie]:
                    inFaces.append(optie)
            inFaces = set(inFaces)
            print(f"Present in faces: {inFaces}: {[mesh_.faces()[x] for x in inFaces]}")
            keyPoint = vc[0]
            for inF in inFaces:
                if inF not in facesToRemove:
                    facesToRemove.append(inF)
                    if not keyPoint in mesh_.faces()[inF]:
                        print(f"Currently looking at face {inF}: {mesh_.faces()[inF]}")
                        newPoint = [p for p in mesh_.faces()[inF] if not p in lijn]
                        print(f"Keypoint: {keyPoint}, newpoint: {newPoint}")
                        for lop in lijn:
                            # print(f"Face between: {keyPoint}, {newPoint[0]} and {lop}")
                            faceTA = [keyPoint, newPoint[0], lop] if inF in dict_[(newPoint[0], lop)] else [keyPoint, lop, newPoint[0]]
                            print(f"The new face will be {faceTA}")
                            facesToAdd.append(faceTA)
    print(f"Problematic: {prob}")
    print(f"The faces to add are: {facesToAdd}")
    print(f"And to remove: {facesToRemove} <> {[mesh_.faces()[x] for x in facesToRemove]}")
    return facesToAdd, facesToRemove, prob


# Update the mesh, remove certain faces and add other faces. For fixing protrusions
def updateMesh(facesToAdd: list, facesToRemove: list, dict_: dict, mesh_: vedo.mesh.Mesh, prob: set):
    print("\nUpdating the dict_")
    opties = [0, 1, 2, 0]
    for ftr in facesToRemove:
        face = mesh_.faces()[ftr]
        print(f"Face to Remove: {ftr} >> {face}")
        for i in range(0, 3):
            dict_[(face[opties[i]], face[opties[i+1]])].remove(ftr)
            if len(dict_[(face[opties[i]], face[opties[i+1]])]) < 1:
                print(f"Delete: {(face[opties[i]], face[opties[i+1]])}")
                del dict_[(face[opties[i]], face[opties[i+1]])]
    extraToRemove = []
    print(f"ExtraToRemove: {extraToRemove}")
    for edge in prob:
        try:
            a = dict_[edge]
            if len(a) > 1:
                print(f"Edge {edge} has {a}")
                extraToRemove.extend(dict_[(edge[0], edge[1])])
                extraToRemove.extend(dict_[(edge[1], edge[0])])
        except KeyError as err:
            pass
    print(f"Extra to remove: {extraToRemove}")
    facesToRemove.extend(extraToRemove)
    toRemove = list(set(facesToRemove))
    toRemove.sort()
    newFacesList, lN = removeAndAddFaces(toRemove, facesToAdd, mesh_)
    # for fta in facesToAdd:
    for index in range(lN, len(newFacesList)):
        fta = newFacesList[index]
        # print(f"Updatemesh: fta: {index} >> {fta}")
        for i in range(0, 3):
            try:
                dict_[(fta[opties[i]], fta[opties[i+1]])].append(index)
            except KeyError as err:
                dict_[(fta[opties[i]], fta[opties[i+1]])] = [index]
    print("Done")
    return newFacesList


# Remove and add certain faces to the mesh
def removeAndAddFaces(toRemove1: list, toAdd: list, mesh_: vedo.mesh.Mesh):
    toRemove = [-1]
    toRemove.extend(toRemove1)
    facesOld = [x for x in mesh_.faces()]
    print(f"ToRemove: {toRemove}, toAdd: {toAdd}, len before {len(facesOld)}")
    faces = removeFaces(toRemove, facesOld)
    lN = len(faces)
    if len(toAdd) > 0:
        faces.extend(toAdd)
    return faces, lN


# Remove faces from the mesh
def removeFaces(toRemove: list, faces: list):
    ans = []
    einde = toRemove[1] if len(toRemove) > 1 else len(faces)
    for i in range(toRemove[0]+1, einde):
        ans.append(faces[i])
    if len(toRemove) > 1:
        ans2 = removeFaces(toRemove[1:], faces)
        if len(ans2) > 0:
            ans.extend(ans2)
    return ans


# Fix the mesh: orientations and holes.
def fixMesh(mesh_: vedo.mesh.Mesh):
    dict_, dubbele, problematic = makeMeshLookup(mesh_)
    # print(f"Number of faces: {len(faces)}, vertices: {len(vertices)}, edges: {len(dict_.keys())} ==> doubles: {dubbele}")
    # for i in problematic:
    #     print(dict_[(i[0], i[1])])
    #     print(dict_[(i[1], i[0])])

    # print(f"Dict keys: {dict_.keys()} & values {dict_.values()}")
    print(f"Length of faces: {len(mesh_.faces())}")
    facesOld = [f for f in mesh_.faces()]
    facesNew = findWrongFaces2(facesOld, dict_)
    newMesh = vedo.mesh.Mesh([mesh_.vertices(), facesNew])

    # print(f"Dict keys: {dict_.keys()} & values {dict_.values()}")
    problematic = [x for x in problematic if len(dict_[x]) > 1]
    print(f"Problematic is now: {problematic}")
    print(f"Length of faces 2: {len(facesNew)}")

    # Fixing the holes
    facesToAdd, facesToRemove, probN = deleteFalsePoints(newMesh, dict_, problematic)
    print(f"Length of faces 3: {len(newMesh.faces())}")
    if len(facesToAdd) > 1 or len(facesToRemove) > 1 or len(probN) > 1:
        newFacesList = updateMesh(facesToAdd, facesToRemove, dict_, newMesh, problematic)

        newMesh = vedo.mesh.Mesh([newMesh.vertices(), newFacesList])
        print(f"Length of faces 4: {len(newMesh.faces())}")

    lengteBeforeFilling = len(newMesh.faces())
    newMesh.fillHoles(size=0.5)
    print(f"Length before: {lengteBeforeFilling}, after: {len(newMesh.faces())} ==> {newMesh.faces()[lengteBeforeFilling:]}")

    dict_2, dubbele, problematic = makeMeshLookup(newMesh)
    newFaces = findWrongFaces2(newMesh.faces(), dict_2, lengteBeforeFilling, 0)
    newMesh = vedo.mesh.Mesh([newMesh.vertices(), newFaces])

    # for p in probN:
    #     print(f"In dictionary {p}, {dict_2[p]}")
    #     print(f"In dictionary rev {(p[1], p[0])}, {dict_2[(p[1], p[0])]}")
    return newMesh, probN, lengteBeforeFilling, dict_2

    #findAndFixHoles(mesh_, dict_)
    # fixing holes, if len(keys) != 3*len(faces) ==> There must be a hole
    # if len(dict_.keys()) != 3*len(faces):
    #     # stitch holes
    #     print("HOLEs")
    # return mesh_, problematic, dict_


#
#
# Below is only for testing
#
#



def testFixingHoles():
    mesh_ = vedo.load('DB_decim/Human/5.ply')
    _, prob, dict_ = fixMesh(mesh_)
    colorlist = [vedo.getColor(rgb=[0, 0, 255]) for x in mesh_.faces()]
    # for i in range(2671, 2675):
    #     colorlist[i] = vedo.getColor(rgb=[255,0,0])
    # colorlist[1242] = vedo.getColor(rgb=[255, 0, 0])
    # colorlist[1538] = vedo.getColor(rgb=[255, 0, 0])
    # colorlist[1537] = vedo.getColor(rgb=[255, 0, 0])
    alphas = [0.0 for x in mesh_.faces()]
    facesToPaint = []
    for i in prob:
        facesToPaint.extend(dict_[(i[0], i[1])])
        facesToPaint.extend(dict_[(i[1], i[0])])
    print(f"Faces to paint a different colour: {facesToPaint}")
    print(f"Prob: {prob}")
    verticesToCheck = [mesh_.faces()[i] for i in facesToPaint]
    verticesToCheck3 = set([item for i in facesToPaint for item in mesh_.faces()[i]])
    verticesToCheck2 = set([item for sublist in verticesToCheck for item in sublist])
    for i in facesToPaint:
        if 1113 in mesh_.faces()[i]: # 8.ply => 1486
            colorlist[i] = vedo.getColor(rgb=[255, 255, 0])
        # elif 816 in mesh_.faces()[i]:
        #     colorlist[i] = vedo.getColor(rgb=[255, 0, 255])
        # elif 1463 in mesh_.faces()[i]:
        #     colorlist[i] = vedo.getColor(rgb=[0, 255, 255])
        else:
            colorlist[i] = vedo.getColor(rgb=[255, 0, 0])
        # colorlist[i] = vedo.getColor(rgb=[255, 0, 0])
        alphas[i] = 1.0
    print(verticesToCheck)
    print(verticesToCheck3)

    facesToAdd, facesToRemove = deleteFalsePoints(mesh_, dict_, prob)
    newFacesList = updateMesh(facesToAdd, facesToRemove, dict_, mesh_, prob)
    print(f"Len before {len(mesh_.faces())} - {len(facesToRemove)} + {len(facesToAdd)} =? {len(newFacesList)}")
    print(newFacesList[-5:])
    mesh_ = vedo.mesh.Mesh([mesh_.vertices(), newFacesList])

    count = len(mesh_.faces())
    mesh_.fillHoles(size=0.5)
    colorlist = [vedo.getColor(rgb=[0, 0, 255]) for x in mesh_.faces()]
    for i in range(count, len(mesh_.faces())):
        colorlist[i] = vedo.getColor(rgb=[0, 255, 0])
    print(mesh_.isClosed())

    mesh_.cellIndividualColors(colorlist)
    # mesh_.cellIndividualColors(colorlist, alpha=alphas, alphaPerCell=True)
    mesh_.show(axes=8)


def fixHoleWithBarycenter(loop: list, mesh_: vedo.mesh.Mesh, newVertexID: int):
    points = [mesh_.vertices()[i[0]] for i in loop]
    x, y, z = barycenter.findBaryCenter(points=points)
    print(f"Barycenter: {x}, {y}, {z}")
    vertex = np.array([x, y, z])
    faces = []
    for edge in loop:
        faces.append([newVertexID, edge[0], edge[1]])
    return faces, vertex


def fixHoles(loops: list, mesh_: vedo.mesh.Mesh):
    facesToAdd = []
    verticesToAdd = []
    newVertexID = len(mesh_.vertices())
    for loop in loops:
        if len(loop) == 3:
            facesToAdd.append([loop[0][0], loop[1][0], loop[2][0]])
        else:
            fta, vta = fixHoleWithBarycenter(loop, mesh_, newVertexID)
            newVertexID += 1
            facesToAdd.extend(fta)
            verticesToAdd.append(vta)
    return facesToAdd, verticesToAdd


def findAndFixHoles(mesh_: vedo.mesh.Mesh, dict_: dict):
    # loop over the keys in the dict, check if the opposite exists too.
    # if the opposite does not exist, then there is no triangle there, so add it to the list
    print("Find missing")
    missing = findMissingEdges(dict_)
    print(f"Missing: {missing}")
    loops = findLoopsInEdges(missing)
    facesToAdd, verticesToAdd = fixHoles(loops, mesh_)
    #return facesToAdd, verticesToAdd



def testClosed():
    subfolders = [f.path for f in os.scandir('DB_decim') if f.is_dir()]
    notClosed = []
    for subfolder in subfolders:

        for file in os.listdir(subfolder):
            if file.endswith(".off") or file.endswith(".ply"):
                # print(f"File {file}")
                mesh_ = vedo.load(subfolder + '/' + file)

                beforeIsClosed = mesh_.isClosed()
                if not beforeIsClosed:
                    newMesh, probN, lbf, d2 = fixMesh(mesh_)
                    print(f"Before: {beforeIsClosed} and after: {newMesh.isClosed()}")
                    if not newMesh.isClosed():
                        notClosed.append(f"{subfolder}/{file}")
    print(notClosed)



def findWrongFaces(mesh_: vedo.mesh.Mesh, dict_: dict):
    faces = mesh_.faces()
    isCorrect = [False] * len(faces)
    opties = [0, 1, 2, 0]

    toDo = SimpleQueue()
    toDo.put_nowait(0)
    isCorrect[0] = True
    while not toDo.empty():
        current = toDo.get_nowait()
        print(f"Currently checking face {current}: {faces[current]}")

        # Check neighbours using the edges + dict_
        for i in range(0, len(faces[current])):
            keyNeigh = (faces[current][opties[i+1]], faces[current][opties[i]])
            print(f"Looking up: {keyNeigh}")
            # if len(dict_[(faces[current][opties[i]], faces[current][opties[i+1]])]) > 1:
            #     # fix neighbour
            #     neighbourID = findNeighbour(dict_[(faces[current][opties[i]], faces[current][opties[i+1]])], current)
            #     faces[neighbourID] = [faces[neighbourID][0], faces[neighbourID][2], faces[neighbourID][1]]
            # else:
            #     neighbourID = dict_[(faces[current][opties[i+1]], faces[current][opties[i]])][0]
            try:
                neighbourID = dict_[keyNeigh][0]
            except KeyError as err:
                # fix neighbour
                neighbourID = findNeighbour(dict_[(faces[current][opties[i]], faces[current][opties[i+1]])], current)
                if not isCorrect[neighbourID]:
                    faces[neighbourID] = [faces[neighbourID][0], faces[neighbourID][2], faces[neighbourID][1]]
                    print(f"Fixed face {neighbourID} to [{faces[neighbourID][0]}, {faces[neighbourID][2]}, {faces[neighbourID][1]}]")
                    print(faces[neighbourID])

            # print(f"Neighbour ID: {neighbourID} with {faces[neighbourID]}")
            if not isCorrect[neighbourID]:
                toDo.put_nowait(neighbourID)
                isCorrect[neighbourID] = True
    return mesh_


def main():
    # testClosed()
    # mesh_ = vedo.load('testDB/schijfHole.off')
    # mesh_.show(axes=8)

    # Test orientation stuff
    # mesh_ = vedo.load('testDB/schijfWrongOrient.off')
    # mesh_ = vedo.load('testDB/tetrahedronWrong.off')
    # mesh_ = vedo.load('testDB/schijfN1_conc.off')
    # mesh_ = vedo.load('testDB/schijfConcWrong.off')
    # mesh_ = vedo.load('testDB/schijfHole.off')
    # mesh_.scale(0.25)
    # ['DB_decim\\Bearing/347.ply', 'DB_decim\\Bearing/351.ply', 'DB_decim\\Bird/256.ply', 'DB_decim\\Bird/257.ply', 'DB_decim\\Human/20.ply']
    mesh_ = vedo.load('DB_scale/Bird/257.ply')

    # dict_, dubbele, problematic = makeMeshLookup(mesh_)
    # facesOld = [f for f in mesh_.faces()]
    # facesNew = findWrongFaces2(facesOld, dict_)
    # newMesh = vedo.mesh.Mesh([mesh_.vertices(), facesNew])

    # newMesh = mesh_
    beforeIsClosed = mesh_.isClosed()
    newMesh, probN, lbf, d2 = fixMesh(mesh_)
    print(f"Before: {beforeIsClosed} and after: {newMesh.isClosed()}")

    colorlist = [vedo.getColor(rgb=[0, 0, 255]) for x in newMesh.faces()]
    # for i in range(lbf, len(newMesh.faces())):
    #     colorlist[i] = vedo.getColor(rgb=[0, 255, 0])
    # poep = [2951, 2984, 2908, 1050]
    # colorlist[2951] = vedo.getColor(rgb=[255, 0, 0])
    # colorlist[2984] = vedo.getColor(rgb=[255, 255, 0])
    # colorlist[2908] = vedo.getColor(rgb=[255, 0, 255])
    # colorlist[1050] = vedo.getColor(rgb=[0, 255, 0])

    ans1 = []
    ans2 = []
    faces = []
    for key in d2.keys():
        if len(d2[key]) > 1:
            ans2.append([key])
            print(f"Key {key} >> {d2[key]}")
        try:
            b = d2[(key[1], key[0])]
        except KeyError as err:
            print(f"Key {key} has no neighbour")
            ans1.append(key)
            faces.extend(d2[key])
    for f in faces:
        colorlist[f] = vedo.getColor(rgb=[255, 0, 0])
    newMesh.fillHoles(0.5)
    print(f"And now it is? {newMesh.isClosed()}")
    print(faces)
    # faces, ln = removeAndAddFaces(list(set(faces)), [], newMesh)
    # newMesh = vedo.mesh.Mesh([newMesh.vertices(), faces])
    # newMesh.fillHoles(0.5)
    # print(f"And now it is? {newMesh.isClosed()}")

    # newMesh.frontFaceCulling(True)
    newMesh.backFaceCulling(True)
    newMesh.cellIndividualColors(colorlist)
    newMesh.show(axes=8)







# ['Airplane/72.ply', 'Airplane/73.ply', 'Airplane/80.ply', 'Ant/90.ply', 'Armadillo/297.ply', 'Bearing/344.ply', 'Bearing/345.ply', 'Bearing/347.ply', 'Bearing/351.ply',
# 'Bearing/354.ply', 'Bearing/355.ply', 'Bearing/356.ply', 'Bearing/359.ply', 'Bird/256.ply', 'Bird/257.ply', 'Bust/305.ply', 'Bust/311.ply', 'Bust/312.ply', 'Hand/181.ply',
# 'Hand/193.ply', 'Human/18.ply', 'Human/19.ply', 'Human/20.ply', 'Human/5.ply', 'Human/8.ply', 'Mech/322.ply', 'Mech/323.ply', 'Mech/328.ply', 'Mech/329.ply', 'Mech/330.ply',
# 'Mech/333.ply', 'Mech/338.ply', 'Plier/205.ply', 'Plier/212.ply', 'Teddy/163.ply', 'Teddy/166.ply', 'Vase/365.ply', 'Vase/370.ply']
def mainNew():
    # mesh_ = vedo.load('DB_scale/Bird/257.ply')
    mesh_ = vedo.load('DB_scale/Bearing/344.ply')

    dict_, dubbele, problematic = makeMeshLookup(mesh_)
    colorlist = [vedo.getColor(rgb=[0, 0, 255]) for x in mesh_.faces()]
    for prob in problematic:
        for faceIndex in dict_[(prob[0], prob[1])]:
            colorlist[faceIndex] = vedo.getColor(rgb=[255, 0, 0])
        for faceIndex in dict_[(prob[1], prob[0])]:
            colorlist[faceIndex] = vedo.getColor(rgb=[255, 0, 0])

    # for ID, face in enumerate(mesh_.faces()):
        # if 681 in face:
        #     colorlist[ID] = vedo.getColor(rgb=[255, 255, 0])
        # if 827 in face:
        #     colorlist[ID] = vedo.getColor(rgb=[255, 0, 255])
        # if 826 in face:
        #     colorlist[ID] = vedo.getColor(rgb=[0, 255, 255])
    # [1242, 1538], [1537]
    # with 681, 827, 826
    # for prob in [(682, 603)]:
    #     for faceIndex in dict_[(prob[0], prob[1])]:
    #         colorlist[faceIndex] = vedo.getColor(rgb=[0, 255, 0])
    #     for faceIndex in dict_[(prob[1], prob[0])]:
    #         colorlist[faceIndex] = vedo.getColor(rgb=[0, 255, 0])

    print(f"Prob: {problematic}")
    mesh_.cellIndividualColors(colorlist)
    mesh_.show(axes=8)


def mainOld():
    # mesh_ = vedo.load('DB/donut3.off')
    # mesh_ = vedo.load('DB/donut3_1counterClock.off')
    file = 'testDB/tetrahedronMissing.off'
    # file = 'testDB/schijfMissing.off'
    mesh_ = vedo.load(file)
    # makeMeshLookup(mesh_)
    # new_mesh = fixMesh(mesh_)
    # print(f"Face 1: {mesh_.faces()[1]}")
    # print(type(mesh_.faces()[0]))
    testje = [x for x in mesh_.faces()]
    testje.append([1, 3, 2])
    print(testje)
    vertices = np.array([x for x in mesh_.vertices()])
    print(vertices)
    # mesh_ = vedo.mesh.buildPolyData(vertices=vertices, faces=testje)
    #mesh_ = vedo.mesh.Mesh([vertices, testje])


    from sklearn.metrics.pairwise import cosine_similarity
    print(cosine_similarity([[1, 1, 0.99]], [[1, 1, 0.991]]))

    mesh_.frontFaceCulling(True)
    # new_mesh.backFaceCulling(True)
    mesh_.show(axes=8)

if __name__ == '__main__':
    mainNew()