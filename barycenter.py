import vedo
from typing import List
import math
import numpy as np
from decimal import *

#
# This file contains all methods for computing the barycenter of a mesh, both using floats and decimals, decimals is used in the end
#

# Given a list of points (of the mesh), computes the barycenter/average x, y & z values
def findBaryCenter(points: list):
    xSum = 0
    ySum = 0
    zSum = 0
    for vertex in points:
        xSum += vertex[0]
        ySum += vertex[1]
        zSum += vertex[2]
    lengte = len(points)
    # print(f"min&maxs: {minMax}")
    return xSum/lengte, ySum/lengte, zSum/lengte


# given a list of lists with floats, computes the barycenter/average x, y & z values
def findBaryCenterPoint(lijst: List[List[float]]):
    assert all(len(vertex) > 2 for vertex in lijst)
    #assert all(all((isinstance(punt, float) or isinstance(punt, int)) for punt in vertex) for vertex in lijst)
    x, y, z = 0, 0, 0
    for vertex in lijst:
        x += vertex[0]
        y += vertex[1]
        z += vertex[2]
    lengte = len(lijst)
    return x/lengte, y/lengte, z/lengte


# Calculate the area of a triangle given its three sides' sizes, a >= b >= c, using Heron's formula
def calcTriangleArea(a: float, b: float, c: float):
    """Using Heron's formula, the numerical stable version

    DO NOT CHANGE THE FORMULA, DO NOT REMOVE PARENTHESES!
    """
    assert a > 0.0 and b > 0.0 and c > 0.0
    if a < b or a < c or b < c:
        print("ERROR")
    print(f"A: {a}, B: {b}, C: {c} ==>> {(a + (b+c))*(c - (a - b))*(c + (a-b))*(a + (b-c))}")
    print(f"{(a + (b+c))} & {(c - (a - b))} & {(c + (a-b))} & {(a + (b-c))} <><> {c} - {a - b}")
    return 0.25 * math.sqrt( (a + (b+c))*(c - (a - b))*(c + (a-b))*(a + (b-c)) )


# Calculates the area of a triangle using Heron's formula given the sizes of the three sides in decimals.
# a >= b >= c. In case of some precision error, the smallest float that's bigger than 0.0 is used
def calcTriangleAreaDecimal(a: Decimal, b: Decimal, c: Decimal):
    """Using Heron's formula, the numerical stable version

    DO NOT CHANGE THE FORMULA, DO NOT REMOVE PARENTHESES!
    """
    assert a > 0.0 and b > 0.0 and c > 0.0
    a_b = a - b
    pts = [(a + (b+c)), (c - a_b), (c + a_b), (a + (b-c))]
    for i in range(0, len(pts)):
        if pts[i] <= Decimal(0.0):
            pts[i] = Decimal(np.nextafter(np.float64(0.0), 1))
    multipl = pts[0] * pts[1] * pts[2] * pts[3]
    return Decimal(0.25) * multipl.sqrt()


# given two points (p1 & ps) returns the distance between them as a decimal
def decimalPointDist(p1: np.ndarray, p2: np.ndarray):
    squared = ((Decimal(float(p1[0])) - Decimal(float(p2[0])))**2) + ((Decimal(float(p1[1])) - Decimal(float(p2[1])))**2) \
              + ((Decimal(float(p1[2])) - Decimal(float(p2[2])))**2)
    return squared.sqrt()


# Calculates the length between two points as a float
def calcLength(p1: np.ndarray, p2: np.ndarray):
    squared = np.sum((p1-p2)**2, axis=0)
    return math.sqrt(squared)


# Given a face, return the vertices (as the face only contains the ID of the vertices, not the vertices themselves)
def getPoints(face: List, vertices: List):
    ans = []
    for v in face:
        ans.append(vertices[v])
    return ans


# Given a list of points (of a triangle) computes the sizes of the edges of this triangle as floats and sorts them
def getDistances(lijst: List):
    ans = [calcLength(lijst[0], lijst[1]),
           calcLength(lijst[1], lijst[2]),
           calcLength(lijst[0], lijst[2])]
    ans.sort(reverse=True)
    return ans


# Given a list of points (of a triangle) computes the sizes of the edges of this triangle as Decimal and sorts them
def getDistancesDecimal(lijst: List):
    ans = [decimalPointDist(lijst[0], lijst[1]),
           decimalPointDist(lijst[1], lijst[2]),
           decimalPointDist(lijst[0], lijst[2])]
    ans.sort(reverse=True)
    return ans


# Given a mesh, computes the barycenter of the model using decimals(!).
def findBaryCenter2(mesh_: vedo.mesh.Mesh):
    # print(mesh_.faces())
    assert all(len(face) == 3 for face in mesh_.faces())
    getcontext().prec = 40
    faces = mesh_.faces()
    vertices = mesh_.vertices()
    totaalArea = 0.0
    X = 0.0
    Y = 0.0
    Z = 0.0
    for face in faces:
        punten = getPoints(face, vertices)
        printt = "Punten: "
        for punt in punten:
            printt += f"({punt[0]}, {punt[1]}, {punt[2]}); "
        # print(printt)
        distances = getDistancesDecimal(punten)
        area = float(calcTriangleAreaDecimal(distances[0], distances[1], distances[2]))
        totaalArea += area

        x1, y1, z1 = findBaryCenterPoint(punten)
        X += x1*area
        Y += y1*area
        Z += z1*area
        # print(f"The area: {area} and center: {x1}, {y1}, {z1}")
    return X/totaalArea, Y/totaalArea, Z/totaalArea


#
# Used for testing
#
def main():
    # squared = np.sum((p1-p2)**2, axis=0)
#     return math.sqrt(squared)
    # Punten: (0.068753, 0.111947, -0.070289); (0.061381, 0.116203, -0.070289); (0.065067, 0.114075, -0.070289);
    getcontext().prec = 40
    # Punten: (-0.004441, -0.129449, -0.991175); (0.004071, -0.129449, -0.991175); (-0.000185, -0.129449, -0.991175);
    # punten = [np.array([0.068753, 0.111947, -0.070289]), np.array([0.061381, 0.116203, -0.070289]), np.array([0.065067, 0.114075, -0.070289])]
    punten = [np.array([0.261133, 0.001055, 0.850596]), np.array([0.261133, 0.008926, 0.850596]), np.array([0.261133, -0.006816, 0.850596])]
    p1_p2 = (punten[0]-punten[1])
    p1_p2_2 = p1_p2**2
    squared = np.sum(p1_p2_2, axis=0)
    sqrt = math.sqrt(squared)
    sqrt2 = np.sqrt(squared)
    print(p1_p2)
    print(p1_p2_2)
    print(squared)
    print(f"sqrt: {sqrt} vs {sqrt2}")
    deci1 = Decimal(1/7)
    print(f"Verschil: {1/7} <> {deci1}")
    print(f"Verschil: {(1/7)**2} <> {deci1**2} <> {deci1 * deci1}")
    print(f"Verschil2: {math.sqrt((1/7))} <> {math.sqrt(deci1)} <> {np.sqrt(deci1)} <> {deci1.sqrt()}")
    distances = getDistancesDecimal(punten)
    area = calcTriangleAreaDecimal(distances[0], distances[1], distances[2])
    print(area)
    deci = Decimal(0)
    neaft = np.nextafter(np.float64(0.0), 1)
    print(type(neaft))
    print(type(Decimal(neaft)))
    print(Decimal(neaft))
    #print('{0:.328f}'.format(np.nextafter(area, 1)))
    # 1.8418520101820636e-20
    # 1.819376140526913e-20  36
    # 1.8193779811882898e-20 40
    # 1.8193779816044748e-20 48


if __name__ == '__main__':
    main()