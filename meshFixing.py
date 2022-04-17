import vedo
import os
import numpy as np
from queue import SimpleQueue
import barycenter
# from fixMesh import deleteFalsePoints, updateMesh


def makeMeshLookup(mesh_: vedo.mesh.Mesh, printt: bool = False):
    faces = mesh_.faces()
    dict_ = {} # key = edge as tuple, value = face id
    opties = [0, 1, 2, 0]
    dubbele = 0
    problematic = set()
    for id, face in enumerate(faces):
        for i in range(0, len(face)):
            try:
                dict_[(face[opties[i]], face[opties[i+1]])].append(id)
                if printt: print(f"ADDED! {face[opties[i]]} <-> {face[opties[i+1]]}")
                dubbele += 1
                problematic.add((face[opties[i]], face[opties[i+1]]))
            except KeyError as err:
                dict_[(face[opties[i]], face[opties[i+1]])] = [id]
    if printt: print(f"Number of faces: {len(faces)}, vertices: {len(mesh_.vertices())}, edges: {len(dict_.keys())}")
    return dict_, dubbele, problematic


def findNeighbour(lijst: list, current: int, startFrom: int):
    # if len(lijst) > 2:
    #     print(f"ERROR, LIST IS TOO LONG: {lijst}")
    for i in range(0, len(lijst)):
        if lijst[i] != current and lijst[i] >= startFrom:
            return lijst[i]
    return -1


def updateDict(dict_: dict, faceBef: list, opties: list, neighbourID: int, faceN: list, printt: bool = False):
    # Fix faces in dict - First remove the old order V1 - V2, next add the new order V2 - V1
    for i in range(0, 3):
        if len(dict_[(faceBef[opties[i]], faceBef[opties[i + 1]])]) < 2:
            del dict_[(faceBef[opties[i]], faceBef[opties[i + 1]])]
        else:
            dict_[(faceBef[opties[i]], faceBef[opties[i + 1]])].remove(neighbourID)
        try:
            dict_[(faceN[opties[i]], faceN[opties[i + 1]])].append(neighbourID)
        except KeyError as err:
            dict_[(faceN[opties[i]], faceN[opties[i + 1]])] = [neighbourID]
        if printt:
            print(f"Remove edge ({faceBef[opties[i]]}, {faceBef[opties[i + 1]]}) for ({faceN[opties[i]]}, {faceN[opties[i + 1]]})")


def findWrongFaces(faces: list, dict_: dict, startFrom: int = 0, startFace: int = 0, printt: bool = False):
    isCorrect = [False] * len(faces)
    opties = [0, 1, 2, 0]

    toDo = SimpleQueue()
    toDo.put_nowait(startFace)
    isCorrect[0] = True
    fixed = []
    while not toDo.empty():
        current = toDo.get_nowait()
        # Check neighbours using the edges + dict_
        for i in range(0, len(faces[current])):
            keyNeigh = (faces[current][opties[i+1]], faces[current][opties[i]])
            try:
                neighbourID = dict_[keyNeigh][0]
            except KeyError as err:  # fix neighbour
                neighbourID = findNeighbour(dict_[(faces[current][opties[i]], faces[current][opties[i+1]])], current, startFrom)
                # print(f"KeyNeigh: {keyNeigh}, Current: {current} and n ID: {neighbourID} >> {faces[current]} <> {faces[neighbourID]};  dict: {dict_[(faces[current][opties[i]], faces[current][opties[i+1]])]}")
                if neighbourID != -1 and not isCorrect[neighbourID]:
                    faceBef = faces[neighbourID]
                    faces[neighbourID] = [faces[neighbourID][0], faces[neighbourID][2], faces[neighbourID][1]]
                    if printt: print(f"Fixed face {neighbourID} from [{faces[neighbourID][0]}, {faces[neighbourID][2]}, {faces[neighbourID][1]}] to {faces[neighbourID]}")
                    updateDict(dict_, faceBef, opties, neighbourID, faces[neighbourID])
                    fixed.append(neighbourID)
                elif printt and neighbourID == -1:
                    print(f"NeighbourID is -1: {dict_[(faces[current][opties[i]], faces[current][opties[i+1]])]}, current: {current}")

            # print(f"Neighbour ID: {neighbourID}")
            if neighbourID != -1 and not isCorrect[neighbourID]:
                toDo.put_nowait(neighbourID)
                isCorrect[neighbourID] = True
    fixed.sort()
    if printt: print(f"Fixed faces: {fixed}")
    return faces


def checkAndCorrectFaceOrder(mesh_: vedo.mesh.Mesh):
    # Construct the dictionary with the Edge -> Face key-value pairs
    dict_, _, _ = makeMeshLookup(mesh_)

    # Check and correct if all faces are oriented consistently
    facesOld = [f for f in mesh_.faces()]
    facesNew = findWrongFaces(facesOld, dict_)
    newMesh = vedo.mesh.Mesh([mesh_.vertices(), facesNew])
    return newMesh


"""
This function still might need some work later. It is quite slow.
"""
def fixHoles(mesh_: vedo.mesh.Mesh):
    # Construct the dictionary with the Edge -> Face key-value pairs
    dict_, dubbele, problematic = makeMeshLookup(mesh_)

    # Fix weird protrusion in the mesh
    facesToAdd, facesToRemove, probN = deleteFalsePoints(mesh_, dict_, problematic)
    if len(facesToAdd) > 1 or len(facesToRemove) > 1 or len(probN) > 1:
        newFacesList = updateMesh(facesToAdd, facesToRemove, dict_, mesh_, problematic)

        # Only way to 'update' a mesh using a new list of faces is as follows:
        newMesh = vedo.mesh.Mesh([mesh_.vertices(), newFacesList])
    else:
        newMesh = mesh_

    # Fill the holes
    newMesh.fillHoles(size=0.5)
    return newMesh


"""
In order to fix the holes in the mesh as well as fixing the orientation of the faces, use this function
"""
def meshFixingFaceOrientationAndHoles(mesh_: vedo.mesh.Mesh):
    # Fixing the holes
    if not mesh_.isClosed():
        mesh_ = fixHoles(mesh_)

    # Fix the faces
    new_mesh = checkAndCorrectFaceOrder(mesh_)
    return new_mesh
