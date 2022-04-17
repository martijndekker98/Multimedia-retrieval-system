import vedo
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import barycenter as bc
from scipy import spatial

def flipping_test(mesh_: vedo.mesh.Mesh):

    new_point_coordinates = []
    centers = mesh_.cellCenters()
    x_centers = centers[:,0]
    y_centers = centers[:,1]
    z_centers = centers[:,2]

    x_c_sum = 0
    y_c_sum = 0
    z_c_sum = 0
    for c in centers:
        x_c = c[0]
        y_c = c[1]
        z_c = c[2]
        x_c_sum += (np.sign(x_c) * (x_c * x_c))
        y_c_sum += (np.sign(y_c) * (y_c * y_c))
        z_c_sum += (np.sign(z_c) * (z_c * z_c))

    points = mesh_.vertices()

    for point in points:
        new_point_x = (np.sign(x_c_sum) * point[0])
        new_point_y = (np.sign(y_c_sum) * point[1])
        new_point_z = (np.sign(z_c_sum) * point[2])

        new_point_coordinates.append((new_point_x, new_point_y, new_point_z))

    mesh_ = mesh_.points(new_point_coordinates)
    return mesh_


def eccentricity(mesh_ : vedo.mesh.Mesh):
    vertices = mesh_.vertices()
    # print(f"First two vertices: \n {vertices[:2]}")
    # print(f"Transposed: \n {vertices[:2].T}")
    covariance_matrix = np.cov(vertices.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # print(f"Eigenvalues: \n {eigenvalues}")

    """
    Eigenvalues (array) is niet altijd gesorteerd, dus [0] is niet altijd de major eigenvalue.
    Vandaar:  """
    eigenvalues = eigenvalues.tolist()
    eigenvalues.sort(reverse=True)
    eccentricity = abs(eigenvalues[0])/eigenvalues[1]
    # print(f"Eccentricity: \n {eccentricity}")
    
    return eccentricity


def get_surface_area(mesh_ : vedo.mesh.Mesh):

    faces = mesh_.faces()
    vertices = mesh_.vertices()

    total_area = 0

    for face in faces:
        points = bc.getPoints(face, vertices)
        distances = bc.getDistancesDecimal(points)
        area = float(bc.calcTriangleAreaDecimal(distances[0], distances[1], distances[2]))
        total_area += area

    # print(total_area, mesh_.area(), total_area - mesh_.area())

    return total_area

def get_bounding_box_volume(mesh_ : vedo.mesh.Mesh):

    bounds = mesh_.bounds()
    xbounds = abs(bounds[0]-bounds[1])
    ybounds = abs(bounds[2]-bounds[3])
    zbounds = abs(bounds[4]-bounds[5])
    volume = xbounds * ybounds * zbounds
    
    return volume

def diameter(mesh_ : vedo.mesh.Mesh):

    max_distance = 0
    vertices = mesh_.vertices()[:1000]
    for vertex_1 in vertices:
        for vertex_2 in vertices:

            distance = bc.decimalPointDist(vertex_1, vertex_2)
            if distance > max_distance:
                max_distance = distance


    return max_distance

# def check_other_orientation(face_1, face_2):




def normal_normalization(mesh_ : vedo.mesh.Mesh): # Oud

    vertices = mesh_.vertices()
    faces = mesh_.faces()

    prev_face = faces[0]
    prev_orientation = 1
    reattach = prev_face
    reattach_orientation = 1
    bump = False

    # counter = 0
    # for face in faces[340:350]:
    #     print(face)


    for face in faces[340:350]:

        shared_vertices = []

        for vertex in range(3):
            if not len(np.where(prev_face == face[vertex])[0]) == 0:
                shared_vertices.append(np.where(prev_face == face[vertex])[0][0])

        if len(shared_vertices) != 2:
            print('Bump')
            prev_face = reattach
            prev_orientation = reattach_orientation
            reattach = face
            # reattach_orientation = orientation
            bump = True
            continue

        print(f"Previous face: {prev_face}")
        print(f"Face: {face}")


        if (shared_vertices[0] - shared_vertices[1]) < 0:
            orientation = -1
        else:
            orientation = 1

        print(f"Orientation: {orientation}")
       



        if orientation == prev_orientation:
            if not bump:
                print(shared_vertices)
                print(orientation, prev_orientation)
                print(f"Previous face: {prev_face}")
                print(f"Face: {face}")
                print('FOUT')



        bump = False
            

        prev_face = face
        prev_orientation = orientation



def new_normal(mesh_ : vedo.mesh.Mesh): # Oud

    vertices = mesh_.vertices()
    faces = mesh_.faces()  
    edges = []
    

    for face in faces:
        edge_1 = [face[0], face[1]]
        edge_2 = [face[1], face[2]]
        edge_3 = [face[2], face[0]]
        edges.append(edge_1)
        edges.append(edge_2)
        edges.append(edge_3)

    rangee = 100
    for i in range(rangee):
        print(i)
        
        for j in range(rangee):
    
            if i == j:
                continue
            edges_to_check = []
            for edge in edges:
                if i in edge and j in edge:
                    edges_to_check.append(edge)

            if len(edges_to_check) == 0:
                continue

            # print(edges_to_check)

            if edges_to_check[0][0] == edges_to_check[1][0]:
                print('Wrong')


def faster_normal(mesh_ : vedo.mesh.Mesh): # Deze functie is de goede

    vertices = mesh_.vertices()
    faces = mesh_.faces()  
    edges = []
    foul_faces = []
    normals = mesh_.normals(cells=True)

    z = 0
    o = 0
    t = 0
    r = 0

    counter = 0
    for face in faces:
        if counter == 0 or counter == 56: # Dit is om een face te manipuleren
            face = [face[0], face[2], face[1]]
        # if face[0] == 1 and face[1] == 0 and face[2] == 2:
        #     face[1] = 2
        #     face[2] = 0

        # if face[0] == 3 and face[1] == 1 and face[2] == 2:
        #     face[1] = 2
        #     face[2] = 1
        edge_1 = [face[0], face[1]]
        edge_2 = [face[1], face[2]]
        edge_3 = [face[2], face[0]]
        edges.append(edge_1)
        edges.append(edge_2)
        edges.append(edge_3)

        counter += 1

    # print(edges)

    for i in range(len(faces)):
        face = faces[i]
        # if face[0] == 1 and face[1] == 0 and face[2] == 2:
        #     face[1] = 2
        #     face[2] = 0
        fouls = 0
        e_1 = face[0]
        e_2 = face[1]
        e_3 = face[2] 
        if ([e_1, e_2] in edges or [e_2, e_1] in edges) and not ([e_1, e_2] in edges and [e_2, e_1] in edges):
            fouls += 1
            # print('1..', e_1, e_2)
        if ([e_2, e_3] in edges or [e_3, e_2] in edges) and not ([e_2, e_3] in edges and [e_3, e_2] in edges):
            fouls += 1 
            # print('2..', e_2, e_3)
        if ([e_3, e_1] in edges or [e_1, e_3] in edges) and not ([e_3, e_1] in edges and [e_1, e_3] in edges):
            fouls += 1
            # print('3..', e_3, e_1)

        if fouls == 0:
            z += 1
        elif fouls == 1:
            o += 1
            # print(face)
        elif fouls == 2:
            t += 1
            # print(face)
        elif fouls == 3:
            r += 1
            # faces[i] = [face[0], face[2], face[1]]
            # print(face)

        if fouls == 1:
            print(f'Surrounding faces of foul face: {face}')

        if fouls == 2 or fouls == 3:
            print(f'Possible foul face: {face, i}')

            foul_faces.append([face, i])

    print('Nr of faces with zero, one, two and three fouls: ', z, o, t, r)

    true_foul_faces = compare_normals(normals, foul_faces, faces)

    print('Number of true foul faces: ', len(true_foul_faces))
    print('The true foul faces (and indices): ', true_foul_faces)

        


def compare_normals(normals, foul_faces, faces):

    foul_face_indices = [j for i,j in foul_faces]
    foul_face_values = [i for i,j in foul_faces]
    foul_face_normals = [normals[i] for i in foul_face_indices]
    true_foul_faces = []

    for i in range(len(foul_face_values)):

        foul_face = foul_face_values[i]
        foul_index = foul_face_indices[i]
        foul_normal = foul_face_normals[i]

        if foul_index-1 < 0:
            surrounding_indices = [foul_index+1, foul_index+2, foul_index+3]
        elif foul_index+2 > len(faces):
            surrounding_indices = [foul_index-1, foul_index-2, foul_index+1]
        else:
            surrounding_indices = [foul_index-1, foul_index+1, foul_index+2]

        surrounding_normals = [normals[j] for j in surrounding_indices]

        cos_sim_1 = 1 - spatial.distance.cosine(foul_normal, surrounding_normals[0])
        cos_sim_2 = 1 - spatial.distance.cosine(foul_normal, surrounding_normals[1])
        cos_sim_3 = 1 - spatial.distance.cosine(foul_normal, surrounding_normals[2])
        check_cos_sim = 1 - spatial.distance.cosine(surrounding_normals[1], surrounding_normals[2])

        print(cos_sim_1, cos_sim_2, cos_sim_3, check_cos_sim)
        if cos_sim_1 < 0.90 or cos_sim_2 < 0.90 or cos_sim_3 < 0.90:
            true_foul_faces.append([foul_face, foul_index])

    return true_foul_faces


        
        

        
    


def subAndSuperSample(mesh_: vedo.mesh.Mesh, desiredNumber: int, boundaries: bool = False, meth: str = 'pro'):
    points = mesh_.polydata().GetNumberOfPoints()
    # print(f"Points: {points}")
    if points > desiredNumber:
        fractie = desiredNumber / points
        # print(f"Fraction is: {fractie}")
        mesh_.decimate(fraction=fractie, method=meth, boundaries=boundaries)

def custom_mesh_loop():
    for item in os.listdir('./testje'):
        # if item == 'donut2corrected.off':
        # if item == 'donut2WRONG.off':
        # if item == 'donut2.off':
        if item == 'block.off':
        # if item == 'tetrahedron.off':

            print(item)
            mesh_ = vedo.load(f'./testje/{item}')
            mesh_.frontFaceCulling(True)
            mesh_.show(axes=8)
            print(mesh_.faces())
            faster_normal(mesh_)



def loop_over_meshes():
    verticesAfter = []
    # subfolders = [f.path for f in os.scandir('DB') if f.is_dir()]
    subfolders = [f.path for f in os.scandir('psb') if f.is_dir()]
    for subfolder in subfolders:
        # if subfolder == 'psb/Ant':
        #     continue
        for file in os.listdir(subfolder):
            if (file.endswith(".off")):
                if file != '92.off':
                    continue
                print('_____________________________')
                print(f"File: {subfolder} / {file}")
                mesh_ = vedo.load(subfolder + '/' + file)
                subAndSuperSample(mesh_, 1500, False, 'pro')
                # mesh_.frontFaceCulling(True)
                # mesh_.show(axes=8)
                # flipping_test(mesh_)
                # print(f'Eccentricity: {eccentricity(mesh_)}')
                # print(f'Surface area: {get_surface_area(mesh_)}')
                # print(f'Bounding box volume: {get_bounding_box_volume(mesh_)}')
                # print(f'Diameter: {diameter(mesh_)}')
                # normal_normalization(mesh_)
                # new_normal(mesh_)
                faster_normal(mesh_)

                sys.exit(1)



def main():
    loop_over_meshes()
    # custom_mesh_loop()


if __name__ == '__main__':
    main()