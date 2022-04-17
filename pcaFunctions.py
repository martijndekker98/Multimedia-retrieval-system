import numpy as np
import vedo
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os


# given the mesh, return lists containg x values, y values and z values
def getXYZlist(mesh_: vedo.mesh.Mesh):
    vertices = mesh_.vertices()
    x, y, z = [], [], []
    for v in vertices:
        x.append(v[0])
        y.append(v[1])
        z.append(v[2])
    return np.array(x), np.array(y), np.array(z)


# Perform PCA on the given mesh
def pca(mesh_: vedo.mesh.Mesh, update: bool = True):
    x, y, z = getXYZlist(mesh_)
    A = np.array([x, y, z])
    A_cov = np.cov(A)
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)

    #print("VERTICES")
    # print(f"Eigenvalues: {eigenvalues}")
    # print(f"Eigenvectors: {eigenvectors}")
    eigenValsIndex = [eigenvalues.tolist().index(i) for i in sorted(eigenvalues.tolist(), reverse=True)]
    # e1 = eigenvectors[:,0] if eigenvalues[0] > eigenvalues[1] else eigenvectors[:,1]
    # e2 = eigenvectors[:,1] if eigenvalues[0] > eigenvalues[1] else eigenvectors[:,0]
    e1 = eigenvectors[:, eigenValsIndex[0]]
    e2 = eigenvectors[:, eigenValsIndex[1]]
    if update:
        e1_e2 = np.cross(e1, e2)
        l = len(mesh_.vertices())
        new_vertices = []
        for i in range(0, l):
            v = mesh_.vertices()[i]
            x = np.dot(v, e1)
            y = np.dot(v, e2)
            z = np.dot(v, e1_e2)
            xyz = np.array([x, y, z], dtype=np.float32)
            new_vertices.append(xyz)
            # print(f"{bef} <>{xyz} <> {mesh_.vertices()[i]} <<>> {type(mesh_.vertices()[i])}")
        return new_vertices
    else:
        return e1, e2


# Calculate the cosine similarity between eigenvectors and the identity vectors and add them
def calculateCosineSimilarityIndentitySummed(e1, e2):
    """Calculate the cosine similarity between eigenvectors and the identity vectors and add them"""
    e1Cor = [1.0, 0.0, 0.0]
    e2Cor = [0.0, 1.0, 0.0]
    cs1 = cosine_similarity([e1], [e1Cor])
    cs2 = cosine_similarity([e2], [e2Cor])
    return (abs(cs1)+abs(cs2))[0][0]

#
#
# Below is for testing only
#
# 

def testPCA():
    distances = []
    subfolders = [f.path for f in os.scandir('DB_decim') if f.is_dir()]
    e1Cor = [1.0, 0.0, 0.0]
    e2Cor = [0.0, 1.0, 0.0]
    for subfolder in subfolders:
        for file in os.listdir(subfolder):
            if (file.endswith(".off") or file.endswith(".ply")):
                mesh_ = vedo.load(subfolder + '/' + file)
                new_vertices = pca(mesh_, True)
                new_mesh = mesh_.points(new_vertices)
                e1, e2 = pca(new_mesh, False)
                # e1, e2 = pca(mesh_, False)
                cs1 = cosine_similarity([e1], [e1Cor])
                cs2 = cosine_similarity([e2], [e2Cor])
                css = (abs(cs1)+abs(cs2))[0][0]
                strt = subfolder.split('\\')[1]
                print(f"Consine similarity: {css} <> {file} >{strt}<")
                distances.append(css)

    # distances = np.random.randn(1000)
    density, bins, _ = plt.hist(distances)
    count, _ = np.histogram(distances, bins)
    binsDiff = bins[1] - bins[0]
    print(binsDiff)
    print(f"Min: {min(distances)} & max: {max(distances)}")
    for x, y, num in zip(bins, density, count):
        if num != 0:
            print(f"add text {num} <> {x}, {y}")
            plt.text(x + (0.2 * binsDiff), y + 1.08, num, fontsize=10, rotation=0)  # x,y,str
            # plt.text(x, y + 1.08, num, fontsize=10, rotation=0)  # x,y,str

    plt.title('Histogram of cosine similarity')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Count')
    print(f"The average is: {np.average(distances)}")
    # plt.text(20,20, 'average nr of vertices: ' + str(avg_vertices) + '        nr of to be refined meshes: ' + str(len(tbr_vertices)))
    # plt.xlim(min(distances), max(distances))
    plt.xlim(1.99999999999996, 2.0)
    plt.show()

def main():
    # Als je een andere hoek wil proberen dan kun je hem eerst laten schrijven en dan lezen en gebruiken
    # mesh_ = vedo.load('DB/cube3trianglesLong.ply')
    # mesh_.rotateZ(85)
    # vedo.write(mesh_, 'DB/cube3trianglesLong85.ply', binary=False)

    mesh_ = vedo.load('DB/cube3trianglesLong85.ply')

    new_vertices = pca(mesh_, True)
    new_mesh = mesh_.points(new_vertices)
    mesh_.show(axes=8)


if __name__ == '__main__':
    #main()
    testPCA()